#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace py = pybind11;

using Array = py::array_t<double, py::array::c_style>;

double cosine_distance(Array a, Array b)
{
  if (a.ndim() != 1 || b.ndim() != 1)
    throw std::invalid_argument("Expected two 1D arrays");
  if (a.shape(0) != b.shape(0))
    throw std::invalid_argument("Array sizes differ");
  if (a.shape(0) == 0)
    throw std::invalid_argument("Empty vectors are not supported");

  auto aa = a.unchecked<1>();
  auto bb = b.unchecked<1>();
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;

  for (py::ssize_t i = 0; i < a.shape(0); ++i) {
    dot += aa(i) * bb(i);
    na += aa(i) * aa(i);
    nb += bb(i) * bb(i);
  }

  if (na == 0.0 || nb == 0.0)
    throw std::invalid_argument("Zero-length norm");
  return 1.0 - dot / (std::sqrt(na) * std::sqrt(nb));
}

py::array_t<double> batch_distance(Array x, Array q)
{
  if (x.ndim() != 2 || q.ndim() != 1)
    throw std::invalid_argument("Expected X[N,D] and q[D]");
  if (x.shape(1) != q.shape(0))
    throw std::invalid_argument("Dimension mismatch");
  if (q.shape(0) == 0)
    throw std::invalid_argument("Empty vectors are not supported");

  const py::ssize_t n = x.shape(0);
  const py::ssize_t d = x.shape(1);
  py::array_t<double> result(n);

  const double* xp = x.data();
  const double* qp = q.data();
  double* rp = result.mutable_data();

  {
    py::gil_scoped_release release;
    double nq = 0.0;
    for (py::ssize_t j = 0; j < d; ++j)
      nq += qp[j] * qp[j];
    if (nq == 0.0)
      throw std::invalid_argument("Zero query vector");

    for (py::ssize_t i = 0; i < n; ++i) {
      double dot = 0.0;
      double nx = 0.0;
      const double* row = xp + i * d;
      for (py::ssize_t j = 0; j < d; ++j) {
        dot += row[j] * qp[j];
        nx += row[j] * row[j];
      }
      if (nx == 0.0)
        throw std::invalid_argument("Zero row vector");
      rp[i] = 1.0 - dot / (std::sqrt(nx) * std::sqrt(nq));
    }
  }
  return result;
}

PYBIND11_MODULE(sader_math, m)
{
  m.doc() = "C++ numerical functions for Python ML services";
  m.def("cosine_distance", &cosine_distance,
        py::arg("a").noconvert(), py::arg("b").noconvert(),
        "Cosine distance between two contiguous float64 vectors");
  m.def("batch_distance", &batch_distance,
        py::arg("x").noconvert(), py::arg("q").noconvert(),
        "Cosine distance from each row of X to q");
}
