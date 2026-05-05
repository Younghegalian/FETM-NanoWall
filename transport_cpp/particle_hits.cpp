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

enum Face : int {
    X_MINUS = 0,
    X_PLUS = 1,
    Y_MINUS = 2,
    Y_PLUS = 3,
    Z_MINUS = 4,
    Z_PLUS = 5,
};

enum class TraceKind {
    Free,
    Hit,
    Escaped,
};

struct TraceResult {
    TraceKind kind = TraceKind::Free;
    double distance_um = 0.0;
    Vec3 hit_pos_vox{0.0, 0.0, 0.0};
    Vec3 normal{0.0, 0.0, 0.0};
    int solid_x = -1;
    int solid_y = -1;
    int solid_z = -1;
    int face = -1;
};

struct Particle {
    Vec3 pos_vox;
    Vec3 vel_um_s;
    double next_bg_collision_s = 0.0;
};

static inline size_t index_zyx(int x, int y, int z, int nx, int ny) {
    return (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y)) *
               static_cast<size_t>(nx) +
           static_cast<size_t>(x);
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

static inline Vec3 scale(Vec3 v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}

static inline Vec3 add(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 normalize(Vec3 v) {
    double n = norm(v);
    if (n <= 0.0) {
        return {0.0, 0.0, 1.0};
    }
    return scale(v, 1.0 / n);
}

static std::vector<uint8_t> read_mask(const std::string& path, size_t n) {
    std::vector<uint8_t> data(n);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open mask file");
    }
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n));
    if (static_cast<size_t>(in.gcount()) != n) {
        throw std::runtime_error("mask file has unexpected size");
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

static void write_u32(const std::string& path, const std::vector<uint32_t>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open output file");
    }
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(uint32_t))
    );
}

static inline void increment_saturating(uint32_t& value) {
    if (value < std::numeric_limits<uint32_t>::max()) {
        ++value;
    }
}

static bool inside_box(int x, int y, int z, int nx, int ny, int nz) {
    return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
}

static Face face_from_axis_step(int axis, int step) {
    if (axis == 0) {
        return step > 0 ? X_MINUS : X_PLUS;
    }
    if (axis == 1) {
        return step > 0 ? Y_MINUS : Y_PLUS;
    }
    return step > 0 ? Z_MINUS : Z_PLUS;
}

static Face escape_face_from_axis_step(int axis, int step) {
    if (axis == 0) {
        return step > 0 ? X_PLUS : X_MINUS;
    }
    if (axis == 1) {
        return step > 0 ? Y_PLUS : Y_MINUS;
    }
    return step > 0 ? Z_PLUS : Z_MINUS;
}

static Vec3 normal_from_axis_step(int axis, int step) {
    Vec3 n{0.0, 0.0, 0.0};
    double value = step > 0 ? -1.0 : 1.0;
    if (axis == 0) n.x = value;
    if (axis == 1) n.y = value;
    if (axis == 2) n.z = value;
    return n;
}

static TraceResult trace_segment(
    Vec3 pos_vox,
    Vec3 vel_um_s,
    double dt_s,
    const std::vector<uint8_t>& solid,
    int nx,
    int ny,
    int nz,
    double dx_um
) {
    constexpr double eps = 1e-10;
    double speed_um_s = norm(vel_um_s);
    if (speed_um_s <= 0.0 || dt_s <= 0.0) {
        return {};
    }

    Vec3 dir = normalize(vel_um_s);
    double max_dist_vox = speed_um_s * dt_s / dx_um;
    double traveled_vox = 0.0;
    int ix = static_cast<int>(std::floor(pos_vox.x));
    int iy = static_cast<int>(std::floor(pos_vox.y));
    int iz = static_cast<int>(std::floor(pos_vox.z));

    if (!inside_box(ix, iy, iz, nx, ny, nz) || solid[index_zyx(ix, iy, iz, nx, ny)] != 0) {
        TraceResult result;
        result.kind = TraceKind::Escaped;
        return result;
    }

    while (traveled_vox < max_dist_vox) {
        double tx = std::numeric_limits<double>::infinity();
        double ty = std::numeric_limits<double>::infinity();
        double tz = std::numeric_limits<double>::infinity();
        int sx = 0;
        int sy = 0;
        int sz = 0;

        if (dir.x > 0.0) {
            tx = ((static_cast<double>(ix) + 1.0) - pos_vox.x) / dir.x;
            sx = 1;
        } else if (dir.x < 0.0) {
            tx = (static_cast<double>(ix) - pos_vox.x) / dir.x;
            sx = -1;
        }
        if (dir.y > 0.0) {
            ty = ((static_cast<double>(iy) + 1.0) - pos_vox.y) / dir.y;
            sy = 1;
        } else if (dir.y < 0.0) {
            ty = (static_cast<double>(iy) - pos_vox.y) / dir.y;
            sy = -1;
        }
        if (dir.z > 0.0) {
            tz = ((static_cast<double>(iz) + 1.0) - pos_vox.z) / dir.z;
            sz = 1;
        } else if (dir.z < 0.0) {
            tz = (static_cast<double>(iz) - pos_vox.z) / dir.z;
            sz = -1;
        }

        double t_vox = std::min(tx, std::min(ty, tz));
        if (!std::isfinite(t_vox) || t_vox <= 0.0) {
            t_vox = eps;
        }
        if (traveled_vox + t_vox > max_dist_vox) {
            return {};
        }

        pos_vox = add(pos_vox, scale(dir, t_vox));
        traveled_vox += t_vox;

        int hit_axis = 0;
        if (std::abs(t_vox - tx) <= 1e-9) {
            ix += sx;
            hit_axis = 0;
        }
        if (std::abs(t_vox - ty) <= 1e-9) {
            iy += sy;
            if (std::abs(t_vox - tx) > 1e-9) hit_axis = 1;
        }
        if (std::abs(t_vox - tz) <= 1e-9) {
            iz += sz;
            if (std::abs(t_vox - tx) > 1e-9 && std::abs(t_vox - ty) > 1e-9) hit_axis = 2;
        }

        int step = hit_axis == 0 ? sx : (hit_axis == 1 ? sy : sz);
        if (!inside_box(ix, iy, iz, nx, ny, nz)) {
            TraceResult result;
            result.kind = TraceKind::Escaped;
            result.distance_um = traveled_vox * dx_um;
            result.face = static_cast<int>(escape_face_from_axis_step(hit_axis, step));
            return result;
        }

        if (solid[index_zyx(ix, iy, iz, nx, ny)] != 0) {
            TraceResult result;
            result.kind = TraceKind::Hit;
            result.distance_um = traveled_vox * dx_um;
            result.hit_pos_vox = pos_vox;
            result.normal = normal_from_axis_step(hit_axis, step);
            result.solid_x = ix;
            result.solid_y = iy;
            result.solid_z = iz;
            result.face = static_cast<int>(face_from_axis_step(hit_axis, step));
            return result;
        }

        pos_vox = add(pos_vox, scale(dir, eps));
    }

    return {};
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

static Vec3 random_void_position(std::mt19937_64& rng, const std::vector<size_t>& void_indices, int nx, int ny) {
    std::uniform_int_distribution<size_t> pick(0, void_indices.size() - 1);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    size_t idx = void_indices[pick(rng)];
    size_t slice = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    int z = static_cast<int>(idx / slice);
    size_t rem = idx - static_cast<size_t>(z) * slice;
    int y = static_cast<int>(rem / static_cast<size_t>(nx));
    int x = static_cast<int>(rem - static_cast<size_t>(y) * static_cast<size_t>(nx));
    return {static_cast<double>(x) + uni(rng), static_cast<double>(y) + uni(rng), static_cast<double>(z) + uni(rng)};
}

static Vec3 random_boundary_position(
    std::mt19937_64& rng,
    const std::vector<uint8_t>& solid,
    int nx,
    int ny,
    int nz,
    int face
) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    constexpr double eps = 1e-7;
    for (int attempt = 0; attempt < 100000; ++attempt) {
        Vec3 pos{uni(rng) * nx, uni(rng) * ny, uni(rng) * nz};
        if (face == X_MINUS) pos.x = eps;
        if (face == X_PLUS) pos.x = static_cast<double>(nx) - eps;
        if (face == Y_MINUS) pos.y = eps;
        if (face == Y_PLUS) pos.y = static_cast<double>(ny) - eps;
        if (face == Z_MINUS) pos.z = eps;
        if (face == Z_PLUS) pos.z = static_cast<double>(nz) - eps;
        int ix = std::clamp(static_cast<int>(std::floor(pos.x)), 0, nx - 1);
        int iy = std::clamp(static_cast<int>(std::floor(pos.y)), 0, ny - 1);
        int iz = std::clamp(static_cast<int>(std::floor(pos.z)), 0, nz - 1);
        if (solid[index_zyx(ix, iy, iz, nx, ny)] == 0) {
            return pos;
        }
    }
    throw std::runtime_error("failed to sample boundary void position");
}

static void reset_particle(
    Particle& p,
    std::mt19937_64& rng,
    const std::vector<size_t>& init_void_indices,
    int nx,
    int ny,
    double sigma_um_s,
    double lambda_um
) {
    p.pos_vox = random_void_position(rng, init_void_indices, nx, ny);
    p.vel_um_s = random_velocity(rng, sigma_um_s);
    p.next_bg_collision_s = draw_bg_timer(rng, lambda_um, p.vel_um_s);
}

int main(int argc, char** argv) {
    if (argc != 16 && argc != 17) {
        std::cerr << "usage: particle_hits mask.u8 nx ny nz dx_um n_particle steps dt_s warmup_steps sigma_um_s lambda_um seed init_mode top_min_z_um out_dir [escape_reinject_mode]\n";
        return 2;
    }

    std::string mask_path = argv[1];
    int nx = std::stoi(argv[2]);
    int ny = std::stoi(argv[3]);
    int nz = std::stoi(argv[4]);
    double dx_um = std::stod(argv[5]);
    int n_particle = std::stoi(argv[6]);
    int steps = std::stoi(argv[7]);
    double dt_s = std::stod(argv[8]);
    int warmup_steps = std::stoi(argv[9]);
    double sigma_um_s = std::stod(argv[10]);
    double lambda_um = std::stod(argv[11]);
    uint64_t seed = static_cast<uint64_t>(std::stoull(argv[12]));
    std::string init_mode = argv[13];
    double top_min_z_um = std::stod(argv[14]);
    std::string out_dir = argv[15];
    std::string escape_reinject_mode = argc == 17 ? argv[16] : "boundary";

    if (nx <= 0 || ny <= 0 || nz <= 0 || dx_um <= 0.0 || n_particle <= 0 || steps <= 0 || dt_s <= 0.0) {
        std::cerr << "invalid positive parameter\n";
        return 2;
    }
    warmup_steps = std::max(0, std::min(warmup_steps, steps - 1));
    if (sigma_um_s <= 0.0 || lambda_um <= 0.0) {
        std::cerr << "sigma_um_s and lambda_um must be positive\n";
        return 2;
    }
    if (escape_reinject_mode != "boundary" && escape_reinject_mode != "volume_uniform") {
        std::cerr << "escape_reinject_mode must be boundary or volume_uniform\n";
        return 2;
    }

    size_t n_voxel = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    auto solid = read_mask(mask_path, n_voxel);
    std::vector<size_t> void_indices;
    void_indices.reserve(n_voxel);
    for (size_t i = 0; i < n_voxel; ++i) {
        if (solid[i] == 0) {
            void_indices.push_back(i);
        }
    }
    if (void_indices.empty()) {
        std::cerr << "empty void region\n";
        return 3;
    }
    std::vector<size_t> init_void_indices;
    if (init_mode == "uniform") {
        init_void_indices = void_indices;
    } else if (init_mode == "top") {
        init_void_indices.reserve(void_indices.size());
        const size_t slice = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        for (size_t idx : void_indices) {
            int z = static_cast<int>(idx / slice);
            double z_center_um = (static_cast<double>(z) + 0.5) * dx_um;
            if (z_center_um >= top_min_z_um) {
                init_void_indices.push_back(idx);
            }
        }
        if (init_void_indices.empty()) {
            std::cerr << "top init mode has no void voxels at or above top_min_z_um\n";
            return 4;
        }
    } else {
        std::cerr << "init_mode must be uniform or top\n";
        return 2;
    }

    std::mt19937_64 rng(seed);
    std::vector<Particle> particles(static_cast<size_t>(n_particle));
    for (auto& particle : particles) {
        reset_particle(particle, rng, init_void_indices, nx, ny, sigma_um_s, lambda_um);
    }

    std::vector<uint64_t> face_hits(6 * n_voxel, 0);
    std::vector<uint32_t> particle_hit_counts(static_cast<size_t>(n_particle), 0);
    std::vector<uint32_t> particle_escape_counts(static_cast<size_t>(n_particle), 0);
    std::vector<uint32_t> particle_bg_scatter_counts(static_cast<size_t>(n_particle), 0);
    std::vector<uint32_t> particle_stuck_reset_counts(static_cast<size_t>(n_particle), 0);
    std::vector<uint32_t> particle_current_wall_burst(static_cast<size_t>(n_particle), 0);
    std::vector<uint32_t> particle_max_wall_burst_counts(static_cast<size_t>(n_particle), 0);
    uint64_t total_hits = 0;
    uint64_t total_escapes = 0;
    uint64_t total_stuck_resets = 0;
    uint64_t total_bg_scatters = 0;
    constexpr int max_surface_bounces_per_step = 16;
    constexpr double wall_eps_vox = 1e-7;

    std::ofstream curve(out_dir + "/hit_curve.csv");
    curve << "step,time_s,total_hits,collision_rate_s_inv\n";

    for (int step = 0; step < steps; ++step) {
        bool record = step >= warmup_steps;
        for (size_t particle_idx = 0; particle_idx < particles.size(); ++particle_idx) {
            auto& particle = particles[particle_idx];
            double remaining_s = dt_s;
            bool reset_this_step = false;
            int bounces = 0;

            while (remaining_s > 0.0) {
                TraceResult trace = trace_segment(
                    particle.pos_vox,
                    particle.vel_um_s,
                    remaining_s,
                    solid,
                    nx,
                    ny,
                    nz,
                    dx_um
                );

                if (trace.kind == TraceKind::Free) {
                    Vec3 displacement_vox = scale(particle.vel_um_s, remaining_s / dx_um);
                    particle.pos_vox = add(particle.pos_vox, displacement_vox);
                    remaining_s = 0.0;
                } else if (trace.kind == TraceKind::Escaped) {
                    if (escape_reinject_mode == "boundary" && trace.face >= 0) {
                        particle.pos_vox = random_boundary_position(rng, solid, nx, ny, nz, trace.face);
                        particle.vel_um_s = random_velocity(rng, sigma_um_s);
                    } else {
                        reset_particle(particle, rng, init_void_indices, nx, ny, sigma_um_s, lambda_um);
                        reset_this_step = true;
                    }
                    if (record) {
                        increment_saturating(particle_escape_counts[particle_idx]);
                    }
                    particle_current_wall_burst[particle_idx] = 0;
                    ++total_escapes;
                    remaining_s = 0.0;
                } else {
                    if (record) {
                        size_t solid_idx = index_zyx(trace.solid_x, trace.solid_y, trace.solid_z, nx, ny);
                        face_hits[static_cast<size_t>(trace.face) * n_voxel + solid_idx] += 1;
                        increment_saturating(particle_hit_counts[particle_idx]);
                        increment_saturating(particle_current_wall_burst[particle_idx]);
                        particle_max_wall_burst_counts[particle_idx] = std::max(
                            particle_max_wall_burst_counts[particle_idx],
                            particle_current_wall_burst[particle_idx]
                        );
                        ++total_hits;
                    }

                    double speed = std::max(norm(particle.vel_um_s), 1e-30);
                    double hit_dt = trace.distance_um / speed;
                    remaining_s = std::max(0.0, remaining_s - hit_dt);
                    particle.pos_vox = add(trace.hit_pos_vox, scale(trace.normal, wall_eps_vox));
                    particle.vel_um_s = lambertian_velocity(rng, trace.normal, sigma_um_s);

                    ++bounces;
                    if (bounces >= max_surface_bounces_per_step) {
                        reset_particle(particle, rng, init_void_indices, nx, ny, sigma_um_s, lambda_um);
                        ++total_stuck_resets;
                        reset_this_step = true;
                        if (record) {
                            increment_saturating(particle_stuck_reset_counts[particle_idx]);
                        }
                        particle_current_wall_burst[particle_idx] = 0;
                        remaining_s = 0.0;
                    }
                }
            }

            if (!reset_this_step) {
                particle.next_bg_collision_s -= dt_s;
                if (particle.next_bg_collision_s <= 0.0) {
                    particle.vel_um_s = random_velocity(rng, sigma_um_s);
                    particle.next_bg_collision_s = draw_bg_timer(rng, lambda_um, particle.vel_um_s);
                    if (record) {
                        increment_saturating(particle_bg_scatter_counts[particle_idx]);
                    }
                    particle_current_wall_burst[particle_idx] = 0;
                    ++total_bg_scatters;
                }
            }
        }

        if (record) {
            double elapsed = static_cast<double>(step - warmup_steps + 1) * dt_s;
            double rate = elapsed > 0.0 ? static_cast<double>(total_hits) / elapsed : 0.0;
            curve << (step + 1) << "," << std::setprecision(17) << elapsed << "," << total_hits << "," << rate << "\n";
        }
    }

    write_u64(out_dir + "/face_hits.u64", face_hits);
    write_u32(out_dir + "/particle_hit_counts.u32", particle_hit_counts);
    write_u32(out_dir + "/particle_escape_counts.u32", particle_escape_counts);
    write_u32(out_dir + "/particle_bg_scatter_counts.u32", particle_bg_scatter_counts);
    write_u32(out_dir + "/particle_stuck_reset_counts.u32", particle_stuck_reset_counts);
    write_u32(out_dir + "/particle_max_wall_burst_counts.u32", particle_max_wall_burst_counts);

    double simulated_time_s = static_cast<double>(steps - warmup_steps) * dt_s;
    double collision_rate_s_inv = simulated_time_s > 0.0 ? static_cast<double>(total_hits) / simulated_time_s : 0.0;
    std::cout << std::setprecision(17);
    std::cout << "nx=" << nx << "\n";
    std::cout << "ny=" << ny << "\n";
    std::cout << "nz=" << nz << "\n";
    std::cout << "n_voxel=" << n_voxel << "\n";
    std::cout << "n_void=" << void_indices.size() << "\n";
    std::cout << "n_init_void=" << init_void_indices.size() << "\n";
    std::cout << "n_particle=" << n_particle << "\n";
    std::cout << "steps=" << steps << "\n";
    std::cout << "warmup_steps=" << warmup_steps << "\n";
    std::cout << "dt_s=" << dt_s << "\n";
    std::cout << "simulated_time_s=" << simulated_time_s << "\n";
    std::cout << "total_hits=" << total_hits << "\n";
    std::cout << "collision_rate_s_inv=" << collision_rate_s_inv << "\n";
    std::cout << "total_escapes=" << total_escapes << "\n";
    std::cout << "total_stuck_resets=" << total_stuck_resets << "\n";
    std::cout << "total_bg_scatters=" << total_bg_scatters << "\n";
    std::cout << "sigma_um_s=" << sigma_um_s << "\n";
    std::cout << "lambda_um=" << lambda_um << "\n";
    std::cout << "init_mode=" << init_mode << "\n";
    std::cout << "escape_reinject_mode=" << escape_reinject_mode << "\n";
    std::cout << "top_min_z_um=" << top_min_z_um << "\n";
    return 0;
}
