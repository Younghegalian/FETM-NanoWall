#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

struct Vec3 {
    double x;
    double y;
    double z;
};

struct RayResult {
    bool hit = false;
    bool escaped = false;
    bool lost = false;
    double distance_um = 0.0;
};

struct ThreadLocal {
    std::vector<double> phi_scatter;
    std::vector<double> phi_surface;
    double escape_mass = 0.0;
    double lost_mass = 0.0;
    double source_mass_total = 0.0;

    explicit ThreadLocal(size_t n) : phi_scatter(n, 0.0), phi_surface(n, 0.0) {}
};

static inline size_t index_zyx(int x, int y, int z, int nx, int ny) {
    return (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y)) *
               static_cast<size_t>(nx) +
           static_cast<size_t>(x);
}

static std::vector<Vec3> fibonacci_sphere(int n) {
    std::vector<Vec3> dirs;
    dirs.reserve(static_cast<size_t>(n));
    const double golden_angle = M_PI * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < n; ++i) {
        double z = 1.0 - (2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n));
        double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        double theta = golden_angle * static_cast<double>(i);
        dirs.push_back({std::cos(theta) * r, std::sin(theta) * r, z});
    }
    return dirs;
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

static void write_f32(const std::string& path, const std::vector<double>& values) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<float>(values[i]);
    }
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
}

static RayResult trace_one(
    int sx,
    int sy,
    int sz,
    Vec3 dir,
    const std::vector<uint8_t>& solid,
    int nx,
    int ny,
    int nz,
    double dx_um,
    double lambda_um,
    double max_dist_um,
    int max_reflect,
    bool use_box_reflect,
    double source_mass,
    double dir_weight,
    std::vector<double>& phi_scatter,
    std::vector<double>& phi_surface
) {
    constexpr double eps = 1e-9;
    Vec3 pos{static_cast<double>(sx) + 0.5, static_cast<double>(sy) + 0.5, static_cast<double>(sz) + 0.5};
    int ix = sx;
    int iy = sy;
    int iz = sz;
    int reflect_count = 0;
    double dist_um = 0.0;
    const double weight = source_mass * dir_weight;

    while (true) {
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
            RayResult result;
            result.escaped = true;
            result.distance_um = dist_um;
            return result;
        }
        size_t idx = index_zyx(ix, iy, iz, nx, ny);
        if (solid[idx]) {
            phi_surface[idx] += weight * std::exp(-dist_um / lambda_um);
            RayResult result;
            result.hit = true;
            result.distance_um = dist_um;
            return result;
        }

        double tx = std::numeric_limits<double>::infinity();
        double ty = std::numeric_limits<double>::infinity();
        double tz = std::numeric_limits<double>::infinity();
        if (dir.x > 0) tx = ((static_cast<double>(ix) + 1.0) - pos.x) / dir.x;
        if (dir.x < 0) tx = (static_cast<double>(ix) - pos.x) / dir.x;
        if (dir.y > 0) ty = ((static_cast<double>(iy) + 1.0) - pos.y) / dir.y;
        if (dir.y < 0) ty = (static_cast<double>(iy) - pos.y) / dir.y;
        if (dir.z > 0) tz = ((static_cast<double>(iz) + 1.0) - pos.z) / dir.z;
        if (dir.z < 0) tz = (static_cast<double>(iz) - pos.z) / dir.z;
        double t_vox = std::min(tx, std::min(ty, tz));
        if (!std::isfinite(t_vox) || t_vox <= 0.0) {
            t_vox = eps;
        }

        double next_dist_um = dist_um + t_vox * dx_um;
        if (next_dist_um > max_dist_um) {
            double surv_a = std::exp(-dist_um / lambda_um);
            double surv_b = std::exp(-max_dist_um / lambda_um);
            phi_scatter[idx] += weight * (surv_a - surv_b);
            RayResult result;
            result.lost = true;
            result.distance_um = max_dist_um;
            return result;
        }

        double surv_a = std::exp(-dist_um / lambda_um);
        double surv_b = std::exp(-next_dist_um / lambda_um);
        phi_scatter[idx] += weight * (surv_a - surv_b);

        pos.x += dir.x * t_vox;
        pos.y += dir.y * t_vox;
        pos.z += dir.z * t_vox;
        dist_um = next_dist_um;

        bool boundary = false;
        if (std::abs(t_vox - tx) <= 1e-10) {
            pos.x += dir.x * eps;
            boundary = boundary || pos.x < 0.0 || pos.x >= static_cast<double>(nx);
        }
        if (std::abs(t_vox - ty) <= 1e-10) {
            pos.y += dir.y * eps;
            boundary = boundary || pos.y < 0.0 || pos.y >= static_cast<double>(ny);
        }
        if (std::abs(t_vox - tz) <= 1e-10) {
            pos.z += dir.z * eps;
            boundary = boundary || pos.z < 0.0 || pos.z >= static_cast<double>(nz);
        }

        if (boundary) {
            if (use_box_reflect && reflect_count < max_reflect) {
                if (pos.x < 0.0) {
                    pos.x = eps;
                    dir.x = -dir.x;
                } else if (pos.x >= static_cast<double>(nx)) {
                    pos.x = static_cast<double>(nx) - eps;
                    dir.x = -dir.x;
                }
                if (pos.y < 0.0) {
                    pos.y = eps;
                    dir.y = -dir.y;
                } else if (pos.y >= static_cast<double>(ny)) {
                    pos.y = static_cast<double>(ny) - eps;
                    dir.y = -dir.y;
                }
                if (pos.z < 0.0) {
                    pos.z = eps;
                    dir.z = -dir.z;
                } else if (pos.z >= static_cast<double>(nz)) {
                    pos.z = static_cast<double>(nz) - eps;
                    dir.z = -dir.z;
                }
                reflect_count += 1;
            } else {
                RayResult result;
                result.escaped = true;
                result.distance_um = dist_um;
                return result;
            }
        }

        ix = static_cast<int>(std::floor(pos.x));
        iy = static_cast<int>(std::floor(pos.y));
        iz = static_cast<int>(std::floor(pos.z));
    }
}

int main(int argc, char** argv) {
    if (argc < 12 || argc > 13) {
        std::cerr << "usage: transport_dda mask.u8 nx ny nz dx_um lambda_um n_dir max_dist_factor max_reflect use_box_reflect out_dir [n_thread]\n";
        return 2;
    }
    std::string mask_path = argv[1];
    int nx = std::stoi(argv[2]);
    int ny = std::stoi(argv[3]);
    int nz = std::stoi(argv[4]);
    double dx_um = std::stod(argv[5]);
    double lambda_um = std::stod(argv[6]);
    int n_dir = std::stoi(argv[7]);
    double max_dist_factor = std::stod(argv[8]);
    int max_reflect = std::stoi(argv[9]);
    bool use_box_reflect = std::stoi(argv[10]) != 0;
    std::string out_dir = argv[11];
    int n_thread = 0;
    if (argc == 13) {
        n_thread = std::stoi(argv[12]);
    }
    if (n_thread <= 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        n_thread = static_cast<int>(std::max(1u, std::min(4u, hw == 0 ? 4u : hw)));
    }

    size_t n = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    auto solid = read_mask(mask_path, n);
    size_t n_void = 0;
    for (auto value : solid) {
        if (!value) ++n_void;
    }
    if (n_void == 0) {
        std::cerr << "empty void region\n";
        return 3;
    }

    std::vector<double> accessibility(n, 0.0);
    std::vector<double> vis_ang(n, 0.0);
    std::vector<double> d_min_um(n, -1.0);
    auto dirs = fibonacci_sphere(n_dir);

    const double source_mass = 1.0 / static_cast<double>(n_void);
    const double dir_weight = 1.0 / static_cast<double>(n_dir);
    const double max_dist_um = max_dist_factor * lambda_um;
    std::vector<ThreadLocal> locals;
    locals.reserve(static_cast<size_t>(n_thread));
    for (int t = 0; t < n_thread; ++t) {
        locals.emplace_back(n);
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_thread));

    const size_t chunk_size = 4096;
    std::atomic<size_t> next_idx{0};

    auto process_chunks = [&](int tid) {
        ThreadLocal& local = locals[static_cast<size_t>(tid)];
        const size_t slice = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        while (true) {
            size_t begin = next_idx.fetch_add(chunk_size);
            if (begin >= n) break;
            size_t end = std::min(n, begin + chunk_size);
            for (size_t src_idx = begin; src_idx < end; ++src_idx) {
                if (solid[src_idx]) continue;
                int z = static_cast<int>(src_idx / slice);
                size_t rem = src_idx - static_cast<size_t>(z) * slice;
                int y = static_cast<int>(rem / static_cast<size_t>(nx));
                int x = static_cast<int>(rem - static_cast<size_t>(y) * static_cast<size_t>(nx));
                local.source_mass_total += source_mass;
                int hit_count = 0;
                double access_sum = 0.0;
                double d_min = std::numeric_limits<double>::infinity();
                for (const auto& dir : dirs) {
                    RayResult result = trace_one(
                        x,
                        y,
                        z,
                        dir,
                        solid,
                        nx,
                        ny,
                        nz,
                        dx_um,
                        lambda_um,
                        max_dist_um,
                        max_reflect,
                        use_box_reflect,
                        source_mass,
                        dir_weight,
                        local.phi_scatter,
                        local.phi_surface
                    );
                    if (result.hit) {
                        ++hit_count;
                        d_min = std::min(d_min, result.distance_um);
                        access_sum += std::exp(-result.distance_um / lambda_um);
                    } else if (result.escaped) {
                        local.escape_mass += source_mass * dir_weight * std::exp(-result.distance_um / lambda_um);
                    } else if (result.lost) {
                        local.lost_mass += source_mass * dir_weight * std::exp(-max_dist_um / lambda_um);
                    }
                }
                accessibility[src_idx] = access_sum * dir_weight;
                vis_ang[src_idx] = static_cast<double>(hit_count) * dir_weight;
                if (std::isfinite(d_min)) {
                    d_min_um[src_idx] = d_min;
                }
            }
        }
    };

    for (int t = 0; t < n_thread; ++t) {
        workers.emplace_back(process_chunks, t);
    }
    for (auto& worker : workers) {
        worker.join();
    }

    std::vector<double> phi_scatter(n, 0.0);
    std::vector<double> phi_surface(n, 0.0);
    double escape_mass = 0.0;
    double lost_mass = 0.0;
    double source_mass_total = 0.0;
    for (auto& local : locals) {
        escape_mass += local.escape_mass;
        lost_mass += local.lost_mass;
        source_mass_total += local.source_mass_total;
        for (size_t i = 0; i < n; ++i) {
            phi_scatter[i] += local.phi_scatter[i];
            phi_surface[i] += local.phi_surface[i];
        }
    }

    double scatter_sum = 0.0;
    double surface_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        scatter_sum += phi_scatter[i];
        surface_sum += phi_surface[i];
    }

    write_f32(out_dir + "/phi_scatter.f32", phi_scatter);
    write_f32(out_dir + "/phi_surface.f32", phi_surface);
    write_f32(out_dir + "/accessibility.f32", accessibility);
    write_f32(out_dir + "/vis_ang.f32", vis_ang);
    write_f32(out_dir + "/d_min_um.f32", d_min_um);

    std::cout << std::setprecision(17);
    std::cout << "nx=" << nx << "\n";
    std::cout << "ny=" << ny << "\n";
    std::cout << "nz=" << nz << "\n";
    std::cout << "n_voxel=" << n << "\n";
    std::cout << "n_void=" << n_void << "\n";
    std::cout << "n_thread=" << n_thread << "\n";
    std::cout << "source_mass_total=" << source_mass_total << "\n";
    std::cout << "scatter_sum=" << scatter_sum << "\n";
    std::cout << "surface_sum=" << surface_sum << "\n";
    std::cout << "escape_mass=" << escape_mass << "\n";
    std::cout << "lost_mass=" << lost_mass << "\n";
    std::cout << "probability_sum=" << scatter_sum + surface_sum + escape_mass + lost_mass << "\n";
    return 0;
}
