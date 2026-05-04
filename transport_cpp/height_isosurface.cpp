#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Vertex {
    float x;
    float y;
    float z;
};

struct Corner {
    int xg;
    int yg;
    int zg;
    double x;
    double y;
    double z;
    double phi;
};

static uint64_t corner_key(int x, int y, int z, int nx, int ny) {
    return (static_cast<uint64_t>(z) * static_cast<uint64_t>(ny) + static_cast<uint64_t>(y)) *
               static_cast<uint64_t>(nx) +
           static_cast<uint64_t>(x);
}

static uint64_t edge_key(uint64_t a, uint64_t b) {
    if (a > b) {
        std::swap(a, b);
    }
    return (a << 32) ^ b;
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

template <typename T>
static void write_binary(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open binary output file");
    }
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

static size_t edge_vertex(
    int a,
    int b,
    const Corner corners[8],
    std::vector<Vertex>& vertices,
    std::unordered_map<uint64_t, size_t>& vertex_cache,
    int nx,
    int ny
) {
    const Corner& ca = corners[a];
    const Corner& cb = corners[b];
    uint64_t ka = corner_key(ca.xg, ca.yg, ca.zg, nx, ny);
    uint64_t kb = corner_key(cb.xg, cb.yg, cb.zg, nx, ny);
    uint64_t key = edge_key(ka, kb);
    auto it = vertex_cache.find(key);
    if (it != vertex_cache.end()) {
        return it->second;
    }

    double denom = ca.phi - cb.phi;
    double t = std::abs(denom) < 1e-15 ? 0.5 : ca.phi / denom;
    t = std::clamp(t, 0.0, 1.0);
    Vertex v{
        static_cast<float>(ca.x + t * (cb.x - ca.x)),
        static_cast<float>(ca.y + t * (cb.y - ca.y)),
        static_cast<float>(ca.z + t * (cb.z - ca.z)),
    };
    size_t idx = vertices.size();
    vertices.push_back(v);
    vertex_cache.emplace(key, idx);
    return idx;
}

static void append_face(
    size_t a,
    size_t b,
    size_t c,
    const std::vector<Vertex>& vertices,
    std::vector<int32_t>& faces,
    double& total_area_um2
) {
    const Vertex& va = vertices[a];
    const Vertex& vb = vertices[b];
    const Vertex& vc = vertices[c];
    double ux = static_cast<double>(vb.x) - static_cast<double>(va.x);
    double uy = static_cast<double>(vb.y) - static_cast<double>(va.y);
    double uz = static_cast<double>(vb.z) - static_cast<double>(va.z);
    double vx = static_cast<double>(vc.x) - static_cast<double>(va.x);
    double vy = static_cast<double>(vc.y) - static_cast<double>(va.y);
    double vz = static_cast<double>(vc.z) - static_cast<double>(va.z);
    double nx = uy * vz - uz * vy;
    double ny = uz * vx - ux * vz;
    double nz = ux * vy - uy * vx;
    double area2 = nx * nx + ny * ny + nz * nz;
    if (area2 <= 1e-24) {
        return;
    }
    total_area_um2 += 0.5 * std::sqrt(area2);
    faces.push_back(static_cast<int32_t>(a));
    if (nz < 0.0) {
        faces.push_back(static_cast<int32_t>(c));
        faces.push_back(static_cast<int32_t>(b));
    } else {
        faces.push_back(static_cast<int32_t>(b));
        faces.push_back(static_cast<int32_t>(c));
    }
}

static void polygonize_tet(
    const Corner corners[8],
    const int tet[4],
    std::vector<Vertex>& vertices,
    std::vector<int32_t>& faces,
    std::unordered_map<uint64_t, size_t>& vertex_cache,
    double& total_area_um2,
    int nx,
    int ny
) {
    int inside[4];
    int outside[4];
    int n_inside = 0;
    int n_outside = 0;
    for (int i = 0; i < 4; ++i) {
        int idx = tet[i];
        if (corners[idx].phi >= 0.0) {
            inside[n_inside++] = idx;
        } else {
            outside[n_outside++] = idx;
        }
    }
    if (n_inside == 0 || n_inside == 4) {
        return;
    }
    if (n_inside == 1) {
        size_t p0 = edge_vertex(inside[0], outside[0], corners, vertices, vertex_cache, nx, ny);
        size_t p1 = edge_vertex(inside[0], outside[1], corners, vertices, vertex_cache, nx, ny);
        size_t p2 = edge_vertex(inside[0], outside[2], corners, vertices, vertex_cache, nx, ny);
        append_face(p0, p1, p2, vertices, faces, total_area_um2);
    } else if (n_inside == 3) {
        size_t p0 = edge_vertex(outside[0], inside[0], corners, vertices, vertex_cache, nx, ny);
        size_t p1 = edge_vertex(outside[0], inside[1], corners, vertices, vertex_cache, nx, ny);
        size_t p2 = edge_vertex(outside[0], inside[2], corners, vertices, vertex_cache, nx, ny);
        append_face(p0, p2, p1, vertices, faces, total_area_um2);
    } else {
        size_t p_ac = edge_vertex(inside[0], outside[0], corners, vertices, vertex_cache, nx, ny);
        size_t p_ad = edge_vertex(inside[0], outside[1], corners, vertices, vertex_cache, nx, ny);
        size_t p_bc = edge_vertex(inside[1], outside[0], corners, vertices, vertex_cache, nx, ny);
        size_t p_bd = edge_vertex(inside[1], outside[1], corners, vertices, vertex_cache, nx, ny);
        append_face(p_ac, p_bc, p_ad, vertices, faces, total_area_um2);
        append_face(p_bc, p_bd, p_ad, vertices, faces, total_area_um2);
    }
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: height_isosurface height.f32 nx ny dx_um zmax_um dz_um vertices.f32 faces.i32\n";
        return 2;
    }

    std::string height_path = argv[1];
    int nx = std::stoi(argv[2]);
    int ny = std::stoi(argv[3]);
    double dx_um = std::stod(argv[4]);
    double zmax_um = std::stod(argv[5]);
    double dz_um = std::stod(argv[6]);
    std::string vertices_path = argv[7];
    std::string faces_path = argv[8];

    if (nx < 2 || ny < 2 || dx_um <= 0.0 || zmax_um <= 0.0 || dz_um <= 0.0) {
        std::cerr << "invalid positive parameter\n";
        return 2;
    }

    auto height = read_binary<float>(height_path, static_cast<size_t>(nx) * static_cast<size_t>(ny));
    int nz = static_cast<int>(std::ceil(zmax_um / dz_um));
    const int tets[6][4] = {
        {0, 1, 3, 7},
        {0, 3, 2, 7},
        {0, 2, 6, 7},
        {0, 6, 4, 7},
        {0, 4, 5, 7},
        {0, 5, 1, 7},
    };

    std::vector<Vertex> vertices;
    std::vector<int32_t> faces;
    double total_area_um2 = 0.0;
    vertices.reserve(static_cast<size_t>(nx) * static_cast<size_t>(ny) * 4);
    faces.reserve(static_cast<size_t>(nx) * static_cast<size_t>(ny) * 12);
    std::unordered_map<uint64_t, size_t> vertex_cache;
    vertex_cache.reserve(static_cast<size_t>(nx) * static_cast<size_t>(ny) * 8);

    for (int iy = 0; iy < ny - 1; ++iy) {
        for (int ix = 0; ix < nx - 1; ++ix) {
            double h00 = static_cast<double>(height[static_cast<size_t>(iy) * nx + ix]);
            double h10 = static_cast<double>(height[static_cast<size_t>(iy) * nx + ix + 1]);
            double h01 = static_cast<double>(height[static_cast<size_t>(iy + 1) * nx + ix]);
            double h11 = static_cast<double>(height[static_cast<size_t>(iy + 1) * nx + ix + 1]);
            double h_min = std::min(std::min(h00, h10), std::min(h01, h11));
            double h_max = std::max(std::max(h00, h10), std::max(h01, h11));
            int z_start = std::max(0, static_cast<int>(std::floor(h_min / dz_um)) - 1);
            int z_stop = std::min(nz, static_cast<int>(std::ceil(h_max / dz_um)) + 1);

            for (int iz = z_start; iz < z_stop; ++iz) {
                double z0 = static_cast<double>(iz) * dz_um;
                double z1 = std::min(static_cast<double>(iz + 1) * dz_um, zmax_um);
                if (z1 <= z0) {
                    continue;
                }
                Corner corners[8] = {
                    {ix, iy, iz, (ix + 0.5) * dx_um, (iy + 0.5) * dx_um, z0, h00 - z0},
                    {ix + 1, iy, iz, (ix + 1.5) * dx_um, (iy + 0.5) * dx_um, z0, h10 - z0},
                    {ix, iy + 1, iz, (ix + 0.5) * dx_um, (iy + 1.5) * dx_um, z0, h01 - z0},
                    {ix + 1, iy + 1, iz, (ix + 1.5) * dx_um, (iy + 1.5) * dx_um, z0, h11 - z0},
                    {ix, iy, iz + 1, (ix + 0.5) * dx_um, (iy + 0.5) * dx_um, z1, h00 - z1},
                    {ix + 1, iy, iz + 1, (ix + 1.5) * dx_um, (iy + 0.5) * dx_um, z1, h10 - z1},
                    {ix, iy + 1, iz + 1, (ix + 0.5) * dx_um, (iy + 1.5) * dx_um, z1, h01 - z1},
                    {ix + 1, iy + 1, iz + 1, (ix + 1.5) * dx_um, (iy + 1.5) * dx_um, z1, h11 - z1},
                };
                double phi_min = corners[0].phi;
                double phi_max = corners[0].phi;
                for (int i = 1; i < 8; ++i) {
                    phi_min = std::min(phi_min, corners[i].phi);
                    phi_max = std::max(phi_max, corners[i].phi);
                }
                if (phi_min > 0.0 || phi_max < 0.0) {
                    continue;
                }
                for (const auto& tet : tets) {
                    polygonize_tet(corners, tet, vertices, faces, vertex_cache, total_area_um2, nx, ny);
                }
            }
        }
    }

    std::vector<float> flat_vertices;
    flat_vertices.reserve(vertices.size() * 3);
    for (const auto& v : vertices) {
        flat_vertices.push_back(v.x);
        flat_vertices.push_back(v.y);
        flat_vertices.push_back(v.z);
    }

    write_binary(vertices_path, flat_vertices);
    write_binary(faces_path, faces);

    std::cout << std::setprecision(17);
    std::cout << "n_vertex=" << vertices.size() << "\n";
    std::cout << "n_face=" << faces.size() / 3 << "\n";
    std::cout << "nx=" << nx << "\n";
    std::cout << "ny=" << ny << "\n";
    std::cout << "nz=" << nz << "\n";
    std::cout << "dx_um=" << dx_um << "\n";
    std::cout << "dz_um=" << dz_um << "\n";
    std::cout << "wall_area_um2=" << total_area_um2 << "\n";
    return 0;
}
