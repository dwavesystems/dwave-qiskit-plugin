# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
import itertools

from parameterized import parameterized, param

from qiskit.aqua.operators import X, Y, Z, I, MatrixOp, SummedOp, StateFn
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit.optimization.converters import QuadraticProgramToQubo
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.algorithms.optimization_algorithm import OptimizationResultStatus

import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, DWaveCliqueSampler
from dwave.cloud.exceptions import ConfigFileError

from dwave.plugins.qiskit import DWaveMinimumEigensolver


def create_random_qubo_qp(size=3, min_bias=-1, max_bias=1, seed=None):
    rng = random.Random(seed)

    qp = QuadraticProgram()
    for v in range(size):
        qp.binary_var(str(v))

    qp.minimize(
        linear={i: rng.randint(min_bias, max_bias) for i in range(size)},
        quadratic={(i, j): rng.randint(min_bias, max_bias)
                   for i, j in itertools.combinations(range(size), 2)},
        constant=0.0)
    return qp


class TestMinimumEigensolver(unittest.TestCase):

    @parameterized.expand([
        ("Z", Z, ({0: -1}, {})),
        ("IZ", I^Z, ({0: -1, 1: 0}, {})),
        ("ZZ", Z^Z, ({}, {(0, 1): 1})),
        ("IZZI", I^Z^Z^I, ({0: 0, 1: 0, 2: 0, 3: 0}, {(1, 2): 1})),
        ("IZZ+ZIZ", SummedOp([I^Z^Z, 2 * Z^I^Z]), ({}, {(0, 1): 1, (0, 2): 2})),
        ("Z_matrix", MatrixOp([[1, 0], [0, -1]]), ({0: -1}, {})),
    ])
    def test_operator_conversion(self, name, operator, ising):
        mes = DWaveMinimumEigensolver(operator)
        bqm = dimod.BQM.from_ising(*ising)
        self.assertEqual(mes.operator, operator)
        self.assertEqual(mes.bqm.spin, bqm)

    @parameterized.expand([
        ("ZZZ", Z^Z^Z),
        ("X", X),
    ])
    def test_non_ising_operator(self, name, operator):
        with self.assertRaises(QiskitOptimizationError):
            DWaveMinimumEigensolver(operator)

    def test_interface(self):
        """Expected output format and exceptions."""

        # trivial qp, ground state: [1]
        qp = QuadraticProgram()
        qp.binary_var('x')
        qp.minimize(linear=[-1])
        operator, offset = qp.to_ising()

        # use exact solver as sampler
        sampler = dimod.ExactSolver()
        dwave_mes = DWaveMinimumEigensolver(sampler=sampler)

        self.assertIsNone(dwave_mes.operator)
        self.assertEqual(dwave_mes.aux_operators, [])

        # test no operator provided error
        with self.assertRaises(ValueError):
            dwave_mes.compute_minimum_eigenvalue()

        # manually set operator
        dwave_mes.operator = operator
        result = dwave_mes.compute_minimum_eigenvalue()
        self.assertEqual(result.eigenvalue, -0.5)
        self.assertIsInstance(result.eigenstate, dict)
        self.assertDictEqual(result.eigenstate, {'1': 1})

        # test aux operator
        dwave_mes.aux_operators = [operator]
        result = dwave_mes.compute_minimum_eigenvalue()
        self.assertEqual(len(result.aux_operator_eigenvalues), 1)

        # test getters
        self.assertEqual(dwave_mes.operator, operator)
        self.assertEqual(dwave_mes.aux_operators, [operator])

        # test .run
        result = dwave_mes.run()
        self.assertEqual(result.eigenvalue, -0.5)

    def test_degenerate_ground_states(self):
        # ground states: '0', '1'
        operator = 0 * Z

        # use exact solver as sampler
        dwave_mes = DWaveMinimumEigensolver(sampler=dimod.ExactSolver())

        result = dwave_mes.compute_minimum_eigenvalue(operator)
        self.assertEqual(result.eigenvalue, 0)
        self.assertDictEqual(result.eigenstate, {'0': 1, '1': 1})

    def test_ground_states_returned_only(self):
        # two ground states, six excited states
        operator = SummedOp([I^Z^Z, 2 * Z^I^Z])
        ground_states = ['100', '011']

        # use exact solver as sampler
        dwave_mes = DWaveMinimumEigensolver(sampler=dimod.ExactSolver())

        result = dwave_mes.compute_minimum_eigenvalue(operator)
        self.assertEqual(set(result.eigenstate), set(ground_states))
        self.assertEqual(sum(result.eigenstate.values()), 2)

    def test_aux_operators(self):
        # two ground states: '01', '10'
        operator = Z^Z
        bqm = dimod.BQM.from_ising({}, {(0, 1): 1}).binary

        aux_operators = [I^Z, Z^I]
        aux_bqms = [dimod.BQM.from_ising({0: -1, 1: 0}, {}).binary,
                    dimod.BQM.from_ising({0: 0, 1: -1}, {}).binary]

        # use exact solver as sampler
        dwave_mes = DWaveMinimumEigensolver(sampler=dimod.ExactSolver())
        result = dwave_mes.compute_minimum_eigenvalue(operator, aux_operators)

        # verify conversion to bqm
        self.assertEqual(dwave_mes.bqm, bqm)
        self.assertListEqual(dwave_mes.aux_bqms, aux_bqms)

        # verify aux_operator eigenvalues (-1, +1) and (+1, -1)
        self.assertEqual(sum(result.aux_operator_eigenvalues[0][:,0]), 0)
        self.assertEqual(sum(result.aux_operator_eigenvalues[1][:,0]), 0)


class TestMinimumEigenOptimizerFlow(unittest.TestCase):
    """Test MinimumEigenOptimizer interfaces correctly with
    DWaveMinimumEigensolver. An exact solver is used instead of QPU.
    """

    def test_unique_solution(self):
        # ground state: [1, 1]
        qp = QuadraticProgram()
        qp.binary_var('x')
        qp.binary_var('y')
        qp.minimize(quadratic={'xy': -1})

        dwave_mes = DWaveMinimumEigensolver(sampler=dimod.ExactSolver())
        optimizer = MinimumEigenOptimizer(dwave_mes)
        result = optimizer.solve(qp)

        self.assertEqual(list(result.x), [1.0, 1.0])
        self.assertEqual(result.fval, -1.0)

    def test_multiple_solutions(self):
        # ground states: [0, 0], [0, 1], [1, 0]
        qp = QuadraticProgram()
        qp.binary_var('x')
        qp.binary_var('y')
        qp.minimize(quadratic={'xy': 1})
        ground_states = set(['00', '01', '10'])

        dwave_mes = DWaveMinimumEigensolver(sampler=dimod.ExactSolver())
        optimizer = MinimumEigenOptimizer(dwave_mes)
        result = optimizer.solve(qp)

        self.assertEqual(result.fval, 0.0)

        # verify all ground states are present
        self.assertEqual(len(result.x), qp.get_num_vars())
        self.assertEqual(len(result.samples), len(ground_states))
        self.assertSetEqual(set(sample for sample, _, _ in result.samples), ground_states)

        # verify raw sampleset is accessible
        self.assertEqual(len(result.min_eigen_solver_result.sampleset), 2 ** qp.get_num_vars())


class TestMinimumEigenOptimizerOnDWave(unittest.TestCase):
    """Test MinimumEigenOptimizer works with DWaveMinimumEigensolver backed by
    the default D-Wave QPU.
    """

    @classmethod
    def setUpClass(cls):
        try:
            cls.qpu = DWaveSampler()
        except (ValueError, ConfigFileError):
            raise unittest.SkipTest("no qpu available")

        cls.sampler = EmbeddingComposite(cls.qpu)

    @classmethod
    def tearDownClass(cls):
        cls.qpu.client.close()

    def test_default_sampler(self):
        """QP is solved with the default QPU sampler."""

        qp = create_random_qubo_qp(size=2, seed=123)

        mes = DWaveMinimumEigensolver()
        result = MinimumEigenOptimizer(mes).solve(qp)

        self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

    def test_solver_selection(self):
        """QP is solved on a QPU with Chimera topology and embedded with a clique embedder."""

        qp = create_random_qubo_qp(size=3, seed=123)

        sampler = DWaveCliqueSampler(solver=dict(topology__type='chimera'))
        mes = DWaveMinimumEigensolver(sampler=sampler)
        result = MinimumEigenOptimizer(mes).solve(qp)

        self.assertEqual(result.status, OptimizationResultStatus.SUCCESS)

    def test_sampling_params(self):
        """Sampling parameters are correctly propagated."""

        qp = create_random_qubo_qp(size=2, seed=123)
        num_reads = 1000

        mes = DWaveMinimumEigensolver(sampler=self.sampler, num_reads=num_reads)
        result = MinimumEigenOptimizer(mes).solve(qp)

        total_samples = sum(result.min_eigen_solver_result.sampleset.record.num_occurrences)
        self.assertEqual(total_samples, num_reads)

    @parameterized.expand([(size, ) for size in range(1, 5)])
    def test_random_qp(self, size):
        """QP solutions from NumPy exact solver and DWave MES match."""

        qp = create_random_qubo_qp(size=size, min_bias=-10, max_bias=10, seed=12345)

        # solve with Numpy MES
        numpy_mes = NumPyMinimumEigensolver()
        numpy_result = MinimumEigenOptimizer(numpy_mes).solve(qp)

        # solve with DWave MES
        dwave_mes = DWaveMinimumEigensolver(sampler=self.sampler)
        dwave_result = MinimumEigenOptimizer(dwave_mes).solve(qp)

        # test only energy, actual ground states might differ
        self.assertEqual(numpy_result.fval, dwave_result.fval)

    def test_quadratic_program(self):
        """MEO with DWave MES works for a non-QUBO QP."""

        # Minimize
        #  obj: a + [ - 2 a*b - 2 a*c + 2 b^2 ]/2 + 3
        # Subject To
        #  lin_geq: a + c >= 1
        #  lin_leq: b <= 3
        # Bounds
        #        a <= 7
        #        b <= 7
        #  0 <= c <= 1
        # Solution
        #  a = 7
        #  b = 3
        #  c = 1
        qp = QuadraticProgram()
        qp.integer_var(0, 7, 'a')
        qp.integer_var(0, 7, 'b')
        qp.binary_var('c')
        qp.minimize(constant=3, linear={'a': 1}, quadratic={'ab': -1, 'bb': 1, 'ac': -1})
        qp.linear_constraint(linear={'a': 1, 'c': 1}, sense='>=', rhs=1, name='lin_geq')
        qp.linear_constraint(linear={'b': 1}, sense='<=', rhs=3, name='lin_leq')

        # NOTE: auto penalty in LinearEqualityToPenalty is >100, resulting with
        # big dynamic range of QUBO coefficients. Try with penalty=1
        #qubo = QuadraticProgramToQubo(penalty=1).convert(qp)

        # solve with Numpy MES
        numpy_mes = NumPyMinimumEigensolver()
        numpy_meo = MinimumEigenOptimizer(numpy_mes, penalty=1)
        numpy_result = numpy_meo.solve(qp)

        # solve with DWave MES
        dwave_mes = DWaveMinimumEigensolver(sampler=self.sampler)
        dwave_meo = MinimumEigenOptimizer(dwave_mes, penalty=1)
        dwave_result = dwave_meo.solve(qp)

        # test only energy, actual ground states might differ
        self.assertEqual(numpy_result.fval, dwave_result.fval)
