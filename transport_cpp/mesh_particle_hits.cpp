#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Vec3 {
    double x;
    double y;
    double z;
};

struct Triangle {
    int a;
    int b;
    int c;
    Vec3 normal;
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    double zmin;
    double zmax;
};

struct Hit {
    bool found = false;
    bool escaped = false;
    double t = 1.0;
    Vec3 pos{0.0, 0.0, 0.0};
    Vec3 normal{0.0, 0.0, 1.0};
    size_t face_id = 0;
};

struct Particle {
    Vec3 pos_um;
    Vec3 vel_um_s;
    double next_bg_collision_s = 0.0;
};

struct AccelGrid {
    int nx = 1;
    int ny = 1;
    int nz = 1;
    double bin_size = 1.0;
    double domain_x = 1.0;
    double domain_y = 1.0;
    double domain_z = 1.0;
    std::vector<std::vector<int>> bins;
};

static inline Vec3 add(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 sub(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 scale(Vec3 v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}

static inline double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

static inline double norm(Vec3 v) {
    return std::sqrt(dot(v, v));
}

static inline Vec3 normalize(Vec3 v) {
    double n = norm(v);
    if (n <= 0.0) {
        return {0.0, 0.0, 1.0};
    }
    return scale(v, 1.0 / n);
}

static inline size_t hidx(int x, int y, int nx) {
    return static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

template <typename T>
static std::vector<T> read_binary(const std::string& path, size_t n) {
    std::vector<T> data(n);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open binary input file");
    }
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * sizeof(T)));
    if (static_cast<size_t>(in.gcount()) != n * sizeof(T)) {
        throw std::runtime_error("binary input file has unexpected size");
    }
    return data;
}

static void write_u64(const std::string& path, const std::vector<uint64_t>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open output file");
    }
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(uint64_t))
    );
}

static double height_at(const std::vector<float>& height, double x_um, double y_um, int nx, int ny, double dx_um) {
    double x_grid = x_um / dx_um - 0.5;
    double y_grid = y_um / dx_um - 0.5;
    x_grid = std::clamp(x_grid, 0.0, static_cast<double>(nx - 1));
    y_grid = std::clamp(y_grid, 0.0, static_cast<double>(ny - 1));
    int x0 = std::clamp(static_cast<int>(std::floor(x_grid)), 0, nx - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(y_grid)), 0, ny - 1);
    int x1 = std::min(x0 + 1, nx - 1);
    int y1 = std::min(y0 + 1, ny - 1);
    double wx = x_grid - static_cast<double>(x0);
    double wy = y_grid - static_cast<double>(y0);
    double h00 = static_cast<double>(height[hidx(x0, y0, nx)]);
    double h10 = static_cast<double>(height[hidx(x1, y0, nx)]);
    double h01 = static_cast<double>(height[hidx(x0, y1, nx)]);
    double h11 = static_cast<double>(height[hidx(x1, y1, nx)]);
    return (1.0 - wx) * (1.0 - wy) * h00 + wx * (1.0 - wy) * h10 + (1.0 - wx) * wy * h01 + wx * wy * h11;
}

static std::vector<Triangle> build_triangles(const std::vector<Vec3>& vertices, const std::vector<int32_t>& faces) {
    std::vector<Triangle> triangles;
    triangles.reserve(faces.size() / 3);
    for (size_t i = 0; i < faces.size(); i += 3) {
        Triangle tri;
        tri.a = faces[i];
        tri.b = faces[i + 1];
        tri.c = faces[i + 2];
        Vec3 a = vertices[tri.a];
        Vec3 b = vertices[tri.b];
        Vec3 c = vertices[tri.c];
        tri.normal = normalize(cross(sub(b, a), sub(c, a)));
        if (tri.normal.z < 0.0) {
            tri.normal = scale(tri.normal, -1.0);
            std::swap(tri.b, tri.c);
        }
        a = vertices[tri.a];
        b = vertices[tri.b];
        c = vertices[tri.c];
        tri.xmin = std::min(a.x, std::min(b.x, c.x));
        tri.xmax = std::max(a.x, std::max(b.x, c.x));
        tri.ymin = std::min(a.y, std::min(b.y, c.y));
        tri.ymax = std::max(a.y, std::max(b.y, c.y));
        tri.zmin = std::min(a.z, std::min(b.z, c.z));
        tri.zmax = std::max(a.z, std::max(b.z, c.z));
        triangles.push_back(tri);
    }
    return triangles;
}

static AccelGrid build_accel_grid(
    const std::vector<Triangle>& triangles,
    double domain_x,
    double domain_y,
    double domain_z,
    double bin_size
) {
    AccelGrid grid;
    grid.domain_x = domain_x;
    grid.domain_y = domain_y;
    grid.domain_z = domain_z;
    grid.bin_size = std::max(bin_size, 1e-9);
    grid.nx = std::max(1, static_cast<int>(std::ceil(domain_x / grid.bin_size)));
    grid.ny = std::max(1, static_cast<int>(std::ceil(domain_y / grid.bin_size)));
    grid.nz = std::max(1, static_cast<int>(std::ceil(domain_z / grid.bin_size)));
    grid.bins.resize(static_cast<size_t>(grid.nx) * static_cast<size_t>(grid.ny) * static_cast<size_t>(grid.nz));

    auto bin_x = [&](double x) {
        return std::clamp(static_cast<int>(std::floor(x / grid.bin_size)), 0, grid.nx - 1);
    };
    auto bin_y = [&](double y) {
        return std::clamp(static_cast<int>(std::floor(y / grid.bin_size)), 0, grid.ny - 1);
    };
    auto bin_z = [&](double z) {
        return std::clamp(static_cast<int>(std::floor(z / grid.bin_size)), 0, grid.nz - 1);
    };

    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        int x0 = bin_x(tri.xmin);
        int x1 = bin_x(tri.xmax);
        int y0 = bin_y(tri.ymin);
        int y1 = bin_y(tri.ymax);
        int z0 = bin_z(tri.zmin);
        int z1 = bin_z(tri.zmax);
        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    size_t idx =
                        (static_cast<size_t>(z) * static_cast<size_t>(grid.ny) + static_cast<size_t>(y)) *
                            static_cast<size_t>(grid.nx) +
                        static_cast<size_t>(x);
                    grid.bins[idx].push_back(static_cast<int>(i));
                }
            }
        }
    }
    return grid;
}

static bool intersect_triangle_segment(
    Vec3 p0,
    Vec3 p1,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    Vec3 normal,
    double& t_out
) {
    constexpr double eps = 1e-12;
    Vec3 dir = sub(p1, p0);
    if (dot(normal, dir) >= -eps) {
        return false;
    }

    Vec3 e1 = sub(b, a);
    Vec3 e2 = sub(c, a);
    Vec3 pvec = cross(dir, e2);
    double det = dot(e1, pvec);
    if (std::abs(det) < eps) {
        return false;
    }
    double inv_det = 1.0 / det;
    Vec3 tvec = sub(p0, a);
    double u = dot(tvec, pvec) * inv_det;
    if (u < -eps || u > 1.0 + eps) {
        return false;
    }
    Vec3 qvec = cross(tvec, e1);
    double v = dot(dir, qvec) * inv_det;
    if (v < -eps || u + v > 1.0 + eps) {
        return false;
    }
    double t = dot(e2, qvec) * inv_det;
    if (t <= eps || t > 1.0 + eps) {
        return false;
    }
    t_out = std::clamp(t, 0.0, 1.0);
    return true;
}

static double first_exit_t(Vec3 p0, Vec3 p1, double xmax, double ymax, double zmax) {
    double best = std::numeric_limits<double>::infinity();
    Vec3 d = sub(p1, p0);
    auto check = [&](double p, double dp, double lo, double hi) {
        if (dp < 0.0 && p + dp < lo) best = std::min(best, (lo - p) / dp);
        if (dp > 0.0 && p + dp > hi) best = std::min(best, (hi - p) / dp);
    };
    check(p0.x, d.x, 0.0, xmax);
    check(p0.y, d.y, 0.0, ymax);
    if (d.z > 0.0 && p1.z > zmax) {
        best = std::min(best, (zmax - p0.z) / d.z);
    }
    if (!std::isfinite(best)) {
        return 2.0;
    }
    return std::clamp(best, 0.0, 1.0);
}

static Hit trace_segment(
    Vec3 p0,
    Vec3 p1,
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles,
    const AccelGrid& grid,
    std::vector<uint32_t>& stamps,
    uint32_t& stamp_value,
    double xmax,
    double ymax,
    double zmax
) {
    Hit best;
    double exit_t = first_exit_t(p0, p1, xmax, ymax, zmax);
    double xmin = std::min(p0.x, p1.x);
    double xmax_seg = std::max(p0.x, p1.x);
    double ymin = std::min(p0.y, p1.y);
    double ymax_seg = std::max(p0.y, p1.y);
    double zmin = std::min(p0.z, p1.z);
    double zmax_seg = std::max(p0.z, p1.z);

    auto bin_x = [&](double x) {
        return std::clamp(static_cast<int>(std::floor(x / grid.bin_size)), 0, grid.nx - 1);
    };
    auto bin_y = [&](double y) {
        return std::clamp(static_cast<int>(std::floor(y / grid.bin_size)), 0, grid.ny - 1);
    };
    auto bin_z = [&](double z) {
        return std::clamp(static_cast<int>(std::floor(z / grid.bin_size)), 0, grid.nz - 1);
    };
    int bx0 = bin_x(xmin);
    int bx1 = bin_x(xmax_seg);
    int by0 = bin_y(ymin);
    int by1 = bin_y(ymax_seg);
    int bz0 = bin_z(zmin);
    int bz1 = bin_z(zmax_seg);
    ++stamp_value;
    if (stamp_value == 0) {
        std::fill(stamps.begin(), stamps.end(), 0);
        stamp_value = 1;
    }

    for (int bz = bz0; bz <= bz1; ++bz) {
        for (int by = by0; by <= by1; ++by) {
            for (int bx = bx0; bx <= bx1; ++bx) {
                size_t bin_idx =
                    (static_cast<size_t>(bz) * static_cast<size_t>(grid.ny) + static_cast<size_t>(by)) *
                        static_cast<size_t>(grid.nx) +
                    static_cast<size_t>(bx);
                const auto& bin = grid.bins[bin_idx];
                for (int tri_idx : bin) {
                    size_t i = static_cast<size_t>(tri_idx);
                    if (stamps[i] == stamp_value) {
                        continue;
                    }
                    stamps[i] = stamp_value;
                    const auto& tri = triangles[i];
                    if (tri.xmax < xmin || tri.xmin > xmax_seg || tri.ymax < ymin || tri.ymin > ymax_seg ||
                        tri.zmax < zmin || tri.zmin > zmax_seg) {
                        continue;
                    }
                    double t = 1.0;
                    if (intersect_triangle_segment(
                            p0,
                            p1,
                            vertices[tri.a],
                            vertices[tri.b],
                            vertices[tri.c],
                            tri.normal,
                            t
                        ) && t < best.t) {
                        best.found = true;
                        best.t = t;
                        best.normal = tri.normal;
                        best.face_id = i;
                    }
                }
            }
        }
    }

    if (exit_t <= 1.0 && (!best.found || exit_t < best.t)) {
        Hit result;
        result.escaped = true;
        result.t = exit_t;
        result.pos = add(p0, scale(sub(p1, p0), exit_t));
        return result;
    }
    if (best.found) {
        best.pos = add(p0, scale(sub(p1, p0), best.t));
    }
    return best;
}

static Vec3 random_velocity(std::mt19937_64& rng, double sigma_um_s) {
    std::normal_distribution<double> normal(0.0, sigma_um_s);
    return {normal(rng), normal(rng), normal(rng)};
}

static Vec3 lambertian_velocity(std::mt19937_64& rng, Vec3 normal, double speed_um_s) {
    constexpr double two_pi = 6.2831853071795864769;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r1 = uni(rng);
    double r2 = uni(rng);
    double cos_theta = std::sqrt(r1);
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    double phi = two_pi * r2;
    Vec3 w = normalize(normal);
    Vec3 helper = std::abs(w.z) < 0.999 ? Vec3{0.0, 0.0, 1.0} : Vec3{0.0, 1.0, 0.0};
    Vec3 u = normalize(cross(helper, w));
    Vec3 v = cross(w, u);
    Vec3 dir = add(add(scale(u, sin_theta * std::cos(phi)), scale(v, sin_theta * std::sin(phi))), scale(w, cos_theta));
    return scale(dir, speed_um_s);
}

static double draw_bg_timer(std::mt19937_64& rng, double lambda_um, Vec3 vel_um_s) {
    double speed = std::max(norm(vel_um_s), 1e-30);
    double mean_tau = lambda_um / speed;
    std::exponential_distribution<double> exp_dist(1.0 / mean_tau);
    return exp_dist(rng);
}

static Vec3 sample_position(
    std::mt19937_64& rng,
    const std::vector<float>& height,
    int nx,
    int ny,
    double dx_um,
    double zmax_um,
    const std::string& init_mode,
    double wall_height_um
) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double xmax = static_cast<double>(nx) * dx_um;
    double ymax = static_cast<double>(ny) * dx_um;
    constexpr double eps = 1e-7;
    for (int attempt = 0; attempt < 100000; ++attempt) {
        double x = uni(rng) * xmax;
        double y = uni(rng) * ymax;
        double h = height_at(height, x, y, nx, ny, dx_um);
        double z0 = h + eps;
        if (init_mode == "top") {
            z0 = std::max(z0, wall_height_um);
        }
        if (z0 >= zmax_um) {
            continue;
        }
        double z = z0 + uni(rng) * (zmax_um - z0);
        return {x, y, z};
    }
    throw std::runtime_error("failed to sample a void particle position");
}

static Vec3 sample_boundary_position(
    std::mt19937_64& rng,
    const std::vector<float>& height,
    int nx,
    int ny,
    double dx_um,
    double zmax_um,
    Vec3 out_pos
) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double xmax = static_cast<double>(nx) * dx_um;
    double ymax = static_cast<double>(ny) * dx_um;
    constexpr double eps = 1e-7;
    for (int attempt = 0; attempt < 100000; ++attempt) {
        double x = uni(rng) * xmax;
        double y = uni(rng) * ymax;
        double z = zmax_um - eps;
        if (out_pos.x < 0.0) {
            x = eps;
            z = uni(rng) * zmax_um;
        } else if (out_pos.x >= xmax) {
            x = xmax - eps;
            z = uni(rng) * zmax_um;
        } else if (out_pos.y < 0.0) {
            y = eps;
            z = uni(rng) * zmax_um;
        } else if (out_pos.y >= ymax) {
            y = ymax - eps;
            z = uni(rng) * zmax_um;
        }
        double h = height_at(height, x, y, nx, ny, dx_um);
        if (z > h + eps) {
            return {x, y, z};
        }
    }
    return sample_position(rng, height, nx, ny, dx_um, zmax_um, "uniform", 0.0);
}

static void reset_particle(
    Particle& p,
    std::mt19937_64& rng,
    const std::vector<float>& height,
    int nx,
    int ny,
    double dx_um,
    double zmax_um,
    const std::string& init_mode,
    double wall_height_um,
    double sigma_um_s,
    double lambda_um
) {
    p.pos_um = sample_position(rng, height, nx, ny, dx_um, zmax_um, init_mode, wall_height_um);
    p.vel_um_s = random_velocity(rng, sigma_um_s);
    p.next_bg_collision_s = draw_bg_timer(rng, lambda_um, p.vel_um_s);
}

int main(int argc, char** argv) {
    if (argc != 23 && argc != 24) {
        std::cerr << "usage: mesh_particle_hits height.f32 nx ny dx_um zmax_um vertices.f32 faces.i32 n_vertex n_face domain_x_um domain_y_um bin_size_um n_particle steps dt_s warmup_steps sigma_um_s lambda_um seed init_mode wall_height_um out_dir [curve_interval_steps]\n";
        return 2;
    }

    std::string height_path = argv[1];
    int nx = std::stoi(argv[2]);
    int ny = std::stoi(argv[3]);
    double dx_um = std::stod(argv[4]);
    double zmax_um = std::stod(argv[5]);
    std::string vertices_path = argv[6];
    std::string faces_path = argv[7];
    size_t n_vertex = static_cast<size_t>(std::stoull(argv[8]));
    size_t n_face = static_cast<size_t>(std::stoull(argv[9]));
    double domain_x_um = std::stod(argv[10]);
    double domain_y_um = std::stod(argv[11]);
    double bin_size_um = std::stod(argv[12]);
    int n_particle = std::stoi(argv[13]);
    int steps = std::stoi(argv[14]);
    double dt_s = std::stod(argv[15]);
    int warmup_steps = std::stoi(argv[16]);
    double sigma_um_s = std::stod(argv[17]);
    double lambda_um = std::stod(argv[18]);
    uint64_t seed = static_cast<uint64_t>(std::stoull(argv[19]));
    std::string init_mode = argv[20];
    double wall_height_um = std::stod(argv[21]);
    std::string out_dir = argv[22];
    int curve_interval_steps = argc == 24 ? std::stoi(argv[23]) : 1;

    if (nx < 2 || ny < 2 || dx_um <= 0.0 || zmax_um <= 0.0 || n_vertex == 0 || n_face == 0 ||
        n_particle <= 0 || steps <= 0 || dt_s <= 0.0) {
        std::cerr << "invalid positive parameter\n";
        return 2;
    }
    if (init_mode != "uniform" && init_mode != "top") {
        std::cerr << "init_mode must be uniform or top\n";
        return 2;
    }
    warmup_steps = std::max(0, std::min(warmup_steps, steps - 1));
    curve_interval_steps = std::max(1, curve_interval_steps);

    auto height = read_binary<float>(height_path, static_cast<size_t>(nx) * static_cast<size_t>(ny));
    auto raw_vertices = read_binary<float>(vertices_path, n_vertex * 3);
    auto faces = read_binary<int32_t>(faces_path, n_face * 3);
    std::vector<Vec3> vertices(n_vertex);
    for (size_t i = 0; i < n_vertex; ++i) {
        vertices[i] = {
            static_cast<double>(raw_vertices[3 * i]),
            static_cast<double>(raw_vertices[3 * i + 1]),
            static_cast<double>(raw_vertices[3 * i + 2]),
        };
    }
    raw_vertices.clear();
    raw_vertices.shrink_to_fit();
    auto triangles = build_triangles(vertices, faces);
    faces.clear();
    faces.shrink_to_fit();
    auto accel = build_accel_grid(triangles, domain_x_um, domain_y_um, zmax_um, bin_size_um);
    std::vector<uint32_t> stamps(n_face, 0);
    uint32_t stamp_value = 0;

    std::mt19937_64 rng(seed);
    std::vector<Particle> particles(static_cast<size_t>(n_particle));
    for (auto& p : particles) {
        reset_particle(p, rng, height, nx, ny, dx_um, zmax_um, init_mode, wall_height_um, sigma_um_s, lambda_um);
    }

    std::vector<uint64_t> face_hits(n_face, 0);
    uint64_t total_hits = 0;
    uint64_t total_escapes = 0;
    uint64_t total_deep_resets = 0;
    uint64_t total_stuck_resets = 0;
    constexpr int max_surface_bounces_per_step = 16;
    constexpr double wall_eps_um = 1e-7;

    std::ofstream curve(out_dir + "/hit_curve.csv");
    curve << "step,time_s,total_hits,collision_rate_s_inv\n";

    for (int step = 0; step < steps; ++step) {
        bool record = step >= warmup_steps;
        for (auto& p : particles) {
            double remaining_s = dt_s;
            bool reset_this_step = false;
            int bounces = 0;

            while (remaining_s > 0.0) {
                double local_h = height_at(height, p.pos_um.x, p.pos_um.y, nx, ny, dx_um);
                if (p.pos_um.z <= local_h) {
                    reset_particle(p, rng, height, nx, ny, dx_um, zmax_um, init_mode, wall_height_um, sigma_um_s, lambda_um);
                    ++total_deep_resets;
                    reset_this_step = true;
                    remaining_s = 0.0;
                    break;
                }

                Vec3 p1 = add(p.pos_um, scale(p.vel_um_s, remaining_s));
                Hit hit = trace_segment(
                    p.pos_um,
                    p1,
                    vertices,
                    triangles,
                    accel,
                    stamps,
                    stamp_value,
                    domain_x_um,
                    domain_y_um,
                    zmax_um
                );

                if (hit.escaped) {
                    p.pos_um = sample_boundary_position(rng, height, nx, ny, dx_um, zmax_um, hit.pos);
                    p.vel_um_s = random_velocity(rng, sigma_um_s);
                    p.next_bg_collision_s = draw_bg_timer(rng, lambda_um, p.vel_um_s);
                    ++total_escapes;
                    reset_this_step = true;
                    remaining_s = 0.0;
                } else if (!hit.found) {
                    p.pos_um = p1;
                    remaining_s = 0.0;
                } else {
                    if (record) {
                        face_hits[hit.face_id] += 1;
                        ++total_hits;
                    }
                    remaining_s = std::max(0.0, remaining_s * (1.0 - hit.t));
                    p.pos_um = add(hit.pos, scale(hit.normal, wall_eps_um));
                    p.vel_um_s = lambertian_velocity(rng, hit.normal, sigma_um_s);
                    ++bounces;
                    if (bounces >= max_surface_bounces_per_step) {
                        reset_particle(p, rng, height, nx, ny, dx_um, zmax_um, init_mode, wall_height_um, sigma_um_s, lambda_um);
                        ++total_stuck_resets;
                        reset_this_step = true;
                        remaining_s = 0.0;
                    }
                }
            }

            if (!reset_this_step) {
                p.next_bg_collision_s -= dt_s;
                if (p.next_bg_collision_s <= 0.0) {
                    p.vel_um_s = random_velocity(rng, sigma_um_s);
                    p.next_bg_collision_s = draw_bg_timer(rng, lambda_um, p.vel_um_s);
                }
            }
        }

        if (record && (((step - warmup_steps) % curve_interval_steps == 0) || step == steps - 1)) {
            double elapsed = static_cast<double>(step - warmup_steps + 1) * dt_s;
            double rate = elapsed > 0.0 ? static_cast<double>(total_hits) / elapsed : 0.0;
            curve << (step + 1) << "," << std::setprecision(17) << elapsed << "," << total_hits << "," << rate << "\n";
        }
    }

    write_u64(out_dir + "/surface_face_hits.u64", face_hits);
    double simulated_time_s = static_cast<double>(steps - warmup_steps) * dt_s;
    double collision_rate_s_inv = simulated_time_s > 0.0 ? static_cast<double>(total_hits) / simulated_time_s : 0.0;

    std::cout << std::setprecision(17);
    std::cout << "nx=" << nx << "\n";
    std::cout << "ny=" << ny << "\n";
    std::cout << "n_vertex=" << n_vertex << "\n";
    std::cout << "n_face=" << n_face << "\n";
    std::cout << "accel_nx=" << accel.nx << "\n";
    std::cout << "accel_ny=" << accel.ny << "\n";
    std::cout << "accel_nz=" << accel.nz << "\n";
    std::cout << "accel_bin_size_um=" << bin_size_um << "\n";
    std::cout << "n_particle=" << n_particle << "\n";
    std::cout << "steps=" << steps << "\n";
    std::cout << "warmup_steps=" << warmup_steps << "\n";
    std::cout << "curve_interval_steps=" << curve_interval_steps << "\n";
    std::cout << "dt_s=" << dt_s << "\n";
    std::cout << "simulated_time_s=" << simulated_time_s << "\n";
    std::cout << "total_hits=" << total_hits << "\n";
    std::cout << "collision_rate_s_inv=" << collision_rate_s_inv << "\n";
    std::cout << "total_escapes=" << total_escapes << "\n";
    std::cout << "total_deep_resets=" << total_deep_resets << "\n";
    std::cout << "total_stuck_resets=" << total_stuck_resets << "\n";
    std::cout << "sigma_um_s=" << sigma_um_s << "\n";
    std::cout << "lambda_um=" << lambda_um << "\n";
    std::cout << "init_mode=" << init_mode << "\n";
    std::cout << "wall_height_um=" << wall_height_um << "\n";
    return 0;
}
