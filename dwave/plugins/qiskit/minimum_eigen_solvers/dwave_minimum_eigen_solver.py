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

import logging
from typing import List, Optional, Union, Dict

import numpy as np
import dimod
from dwave.system import DWaveSampler, AutoEmbeddingComposite

from qiskit.aqua.operators import OperatorBase, LegacyBaseOperator, StateFn
from qiskit.aqua.algorithms import MinimumEigensolver, MinimumEigensolverResult
from qiskit.optimization.problems import QuadraticProgram

__all__ = ['DWaveMinimumEigensolver']

logger = logging.getLogger(__name__)


class DWaveMinimumEigensolver(MinimumEigensolver):
    """Obtain ground state(s) of an Ising Hamiltonian using D-Wave's QPU.

    Args:
        operator:
            Ising Hamiltonian qubit operator, with at most 2 Pauli Zs in
            any Pauli term.
        aux_operators:
            Auxiliary operators to be evaluated at each eigenvalue.
        sampler:
            Instantiated D-Wave sampler. Defaults to
            ~dwave.system.AutoEmbeddingComposite`-wrapped
            `~dwave.system.DWaveSampler` over a QPU solver.
        num_reads:
            Number of QPU reads.

    Note:
        Configured access to D-Wave API/Leap is a prerequisite.

    Example:
        # define a simple quadratic program
        qp = QuadraticProgram()
        qp.binary_var('x')
        qp.binary_var('y')
        qp.minimize(linear=[1,-2], quadratic={('x', 'y'): 1})

        # solve it with a minimum eigen optimizer that uses D-Wave QPU
        dwave_mes = DWaveMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(dwave_mes)
        result = optimizer.solve(qp)

    """

    def __init__(self,
                 operator: Union[OperatorBase, LegacyBaseOperator] = None,
                 aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                             LegacyBaseOperator]]]] = None,
                 sampler: dimod.Sampler = None,
                 num_reads: int = 100,
                 ) -> None:
        super().__init__()
        self.operator = operator
        self.aux_operators = aux_operators

        self._sampler = sampler
        self._num_reads = num_reads

    def supports_aux_operators(self) -> bool:
        # NOTE: needed because of overly strict check on MinimumEigenOptimizer
        # init (see: https://github.com/Qiskit/qiskit-aqua/issues/1306)
        return True

    def _operator_to_bqm(self, operator):
        """Convert an Ising Hamiltonian operator (with at most 2 Pauli Zs in
        any Pauli term) to a `~dimod.BinaryQuadraticModel` suitable for
        submission to a D-Wave sampler.
        """
        # convert `operator` to QUBO, failing with `QiskitOptimizationError`
        # for unsupported operators
        qp = QuadraticProgram()
        qp.from_ising(operator)

        # sanity check
        assert qp.objective.sense is qp.objective.Sense.MINIMIZE

        # construct a BQM
        # (use to_array for linear coefficients to make sure implied, but not
        # used, variables are included)
        return dimod.AdjVectorBQM(qp.objective.linear.to_array(),
                                  qp.objective.quadratic.to_dict(),
                                  qp.objective.constant,
                                  vartype=dimod.BINARY)

    @property
    def operator(self) -> Optional[OperatorBase]:
        return self._operator

    @operator.setter
    def operator(self,
                 operator: Union[OperatorBase, LegacyBaseOperator]) -> None:
        """Convert an Ising Hamiltonian operator to a binary quadratic model
        suitable for submission to a D-Wave sampler.

        Args:
            operator:
                Ising Hamiltonian qubit operator, with at most 2 Pauli Zs in
                any Pauli term.

        Raises:
            QiskitOptimizationError:
                If there are Pauli Xs in any Pauli term, or if there are more
                than 2 Pauli Zs in any Pauli term

        """
        self._operator = operator
        logger.debug('operator set to %r', operator)

        if operator is not None:
            self._bqm = self._operator_to_bqm(operator)
            logger.debug('BQM set to %s', self._bqm)

    @property
    def aux_operators(self) -> Optional[List[Optional[OperatorBase]]]:
        return self._aux_operators

    @aux_operators.setter
    def aux_operators(self,
                      aux_operators: Optional[
                          Union[OperatorBase,
                                LegacyBaseOperator,
                                List[Optional[Union[OperatorBase,
                                                    LegacyBaseOperator]]]]]) -> None:
        if aux_operators is None:
            aux_operators = []
        if not isinstance(aux_operators, list):
            aux_operators = [aux_operators]

        self._aux_operators = aux_operators

    @property
    def bqm(self) -> Optional[dimod.BinaryQuadraticModel]:
        """Binary quadratic model representation of Ising Hamiltonian operator.
        """
        bqm = getattr(self, '_bqm', None)
        if bqm is None:
            raise ValueError('operator not yet set, so bqm not yet available')
        return bqm

    @property
    def aux_bqms(self) -> Optional[List[dimod.BinaryQuadraticModel]]:
        """Binary quadratic model representations of auxiliary Ising Hamiltonian
        operators.
        """
        bqms = getattr(self, '_aux_bqms', None)
        if bqms is None:
            bqms = self._aux_bqms = [self._operator_to_bqm(aux_op) for aux_op in self.aux_operators]
        return bqms

    @property
    def sampler(self) -> dimod.Sampler:
        """Configured D-Wave sampler to use."""
        _sampler = getattr(self, '_sampler', None)
        if _sampler is None:
            _sampler = self._sampler = AutoEmbeddingComposite(DWaveSampler())
        return _sampler

    def compute_minimum_eigenvalue(
            self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: \
                Optional[List[Optional[Union[OperatorBase,
                                             LegacyBaseOperator]]]] = None
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        return self._run()

    def _sample(self) -> dimod.SampleSet:
        params = {}
        if 'num_reads' in self.sampler.parameters:
            params['num_reads'] = self._num_reads
        return self.sampler.sample(self.bqm, **params)

    @staticmethod
    def _stringify(sample: np.ndarray) -> str:
        """Convert numpy.ndarray vector of 0/1 int values to a bit string."""
        return ''.join(map(str, sample))

    def _eval_aux_operators(self, state) -> np.ndarray:
        """Evaluate all aux_operators on the input state."""
        # NOTE: for lack of better specs, we follow the NumPyEigensolver format
        values = [(StateFn(operator, is_measurement=True).eval(state).real, 0)
                  for operator in self._aux_operators]
        return np.array(values, dtype=object)

    def _run(self) -> MinimumEigensolverResult:
        """Sample the Ising Hamiltonian provided on a D-Wave QPU to obtain the
        ground state(s).

        Returns:
            MinimumEigensolverResult:
                Dictionary of results, namely samples in `eigenstate` and
                energies in `eigenvalue`.

        Raises:
            ValueError:
                if no operator has been provided
        """

        sampleset = self._sample()

        logger.debug('sampleset: %r', sampleset)

        if sampleset.vartype is not dimod.BINARY:   # pragma: no cover
            logger.critical("Unexpected result: sampleset=%r", sampleset)
            raise TypeError('expected binary vartype of result sampleset')

        # approximate ground state(s) with lowest-energy samples returned
        ground = sampleset.lowest(rtol=0)
        logger.debug('ground states (%d): %r', len(ground), ground)

        result = MinimumEigensolverResult()
        result.eigenvalue = ground.first.energy

        # NOTE: due to inconsistencies in how DictStateFn values are handled in
        # `MinimumEigenOptimizer` and `DictStateFn` itself (probs vs amplitudes),
        # the safest (for now) is to follow QAOA's approach and return
        # counts/reads per eigenstate (when in superposition).
        result.eigenstate = {self._stringify(rec.sample): rec.num_occurrences
                             for rec in ground.record}

        # optionally, evaluate aux_operators
        if self._aux_operators:
            result.aux_operator_eigenvalues = [self._eval_aux_operators(bitstr)
                                               for bitstr in result.eigenstate]

        # include all samples for inspection
        result.sampleset = sampleset

        logger.debug('run result: %r', result.data)

        return result

    def run(self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: \
                Optional[List[Optional[Union[OperatorBase,
                                             LegacyBaseOperator]]]] = None
    ) -> MinimumEigensolverResult:
        """Obtain ground state(s) of an Ising Hamiltonian using D-Wave's QPU.
        """
        return self.compute_minimum_eigenvalue(operator, aux_operators)
