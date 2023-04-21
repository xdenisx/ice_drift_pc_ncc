#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <boost/program_options.hpp>
#include <boost/functional/hash.hpp>

namespace po = boost::program_options;

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::FT                                                FT;
typedef K::Point_2                                           Point;
typedef K::Segment_2                                         Segment;
typedef CGAL::Alpha_shape_vertex_base_2<K>                   Vb;
typedef CGAL::Alpha_shape_face_base_2<K>                     Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb>          Tds;
typedef CGAL::Delaunay_triangulation_2<K,Tds>                Triangulation_2;
using Edge = Triangulation_2::Edge;
using Vertex = Triangulation_2::Vertex_handle;
typedef CGAL::Alpha_shape_2<Triangulation_2>                 Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator            Alpha_shape_edges_iterator;

template <class OutputIterator>
void alpha_edges( const Alpha_shape_2& A, OutputIterator out)
{
  Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
                             end = A.alpha_shape_edges_end();
  for( ; it!=end; ++it) {
    Edge e = *it;
    auto r0 = A.classify(e.first);
    auto r = A.classify(e);

    *out++ = A.segment(*it);
  }

  auto vit = A.alpha_shape_vertices_begin(),
       vend = A.alpha_shape_vertices_end();
  for( ; vit!=vend; ++vit) {
    Vertex v = *vit;
    auto r = A.classify(v);
  }
}

template <class OutputIterator>
void alpha_verts( const Alpha_shape_2& A, OutputIterator out)
{
  auto vit = A.alpha_shape_vertices_begin(),
       vend = A.alpha_shape_vertices_end();
  for( ; vit!=vend; ++vit) {
    Vertex v = *vit;
    *(++out) = v->point();
  }
}


template <class OutputIterator>
bool file_input(const std::string& in, OutputIterator out)
{
  std::ifstream is(in, std::ios::in);
  if(is.fail())
  {
    std::cerr << "unable to open file for input" << std::endl;
    return false;
  }
  int n;
  is >> n;
  std::cout << "Reading " << n << " points from file" << std::endl;
  CGAL::copy_n(std::istream_iterator<Point>(is), n, out);
  return true;
}

bool save_output(const std::string& out,
                 const std::vector<Segment>& segments,
                 const std::vector<Point>& verts)
{
  std::ofstream os(out, std::ios::out);
  if(os.fail())
  {
    std::cerr << "unable to open file for output" << std::endl;
    return false;
  }

  os << segments.size() << std::endl;
  for (const auto& seg: segments) {
      os << std::fixed << std::setprecision(6) << seg << std::endl;
  }

  os << verts.size() << std::endl;
  for (const auto& v: verts) {
      os << std::fixed << std::setprecision(6) << v << std::endl;
  }

  return true;
}

struct Input {
    std::string in;
    std::string out;
    double alpha;
};

Input parse_input(int argc, char * argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("in", po::value<std::string>()->required(), "input points")
        ("out", po::value<std::string>()->required(), "output edges and verts")
        ("alpha", po::value<double>()->required(), "alpha");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    Input input;
    input.in = vm["in"].as<std::string>();
    input.out = vm["out"].as<std::string>();
    input.alpha = vm["alpha"].as<double>();

    if (input.alpha <= 0)
        throw std::invalid_argument("alpha should be > 0");
    return input;
}

std::vector<Point> filter_verts(const std::vector<Segment>& segments,
                                std::vector<Point>&& verts) {
    struct HashVert {
        size_t operator()(const Point& p) const {
            size_t res = 0;
            boost::hash_combine(res, p.x());
            boost::hash_combine(res, p.y());
            return res;
        }
    };

    std::unordered_set<Point, HashVert> seg_points;
    for (const auto& seg: segments) {
        seg_points.insert(seg.source());
        seg_points.insert(seg.target());
    }

    verts.erase(
        std::remove_if(verts.begin(), verts.end(), [&seg_points](const auto& p) {
            return seg_points.count(p) > 0;
        }),
        verts.end()
    );

    return std::move(verts);
}

// Reads a list of points and returns a list of segments
// corresponding to the Alpha shape.
int main(int argc, char * argv[])
{
  const auto input = parse_input(argc, argv);
  std::list<Point> points;
  if(! file_input(input.in, std::back_inserter(points)))
    return -1;
  Alpha_shape_2 A(points.begin(), points.end(),
                  FT(input.alpha),
                  Alpha_shape_2::GENERAL);

  std::cout<< " Components for alpha " << input.alpha << " " << A.number_of_solid_components() << std::endl;

  std::vector<Segment> segments;
  std::vector<Point> res_points;
  alpha_edges(A, std::back_inserter(segments));
  alpha_verts(A, std::back_inserter(res_points));
  auto filtered_points = filter_verts(segments, std::move(res_points));
  save_output(input.out, segments, filtered_points);

  return 0;
}
