from typing import Any, List, Optional
import logging
import random
import textwrap
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import optuna

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.propose import GroundedProposer
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import (
    create_minibatch,
    create_n_fewshot_demo_sets,
    eval_candidate_program,
    get_program_with_highest_avg_score,
    get_signature,
    print_full_program,
    save_candidate_program,
    set_signature,
)
from .mipro_optimizer_v2 import MIPROv2

logger = logging.getLogger(__name__)

class MIPROv2Plus(MIPROv2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile(
        self,
        student: Any,
        *,
        trainset: List,
        teacher: Any = None,
        valset: Optional[List] = None,
        num_trials: int = 30,
        max_bootstrapped_demos: Optional[int] = None,
        max_labeled_demos: Optional[int] = None,
        seed: Optional[int] = None,
        minibatch: bool = True,
        minibatch_size: int = 25,
        minibatch_full_eval_steps: int = 10,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool = True,
    ) -> Any:
        # Set random seeds
        seed = seed or self.seed
        self._set_random_seeds(seed)

        # Update max demos if specified
        if max_bootstrapped_demos is not None:
            self.max_bootstrapped_demos = max_bootstrapped_demos
        if max_labeled_demos is not None:
            self.max_labeled_demos = max_labeled_demos

        # Set training & validation sets
        trainset, valset = self._set_and_validate_datasets(trainset, valset)

        # Set hyperparameters based on run mode (if set)
        zeroshot_opt = (self.max_bootstrapped_demos == 0) and (
            self.max_labeled_demos == 0
        )
        num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
            student, num_trials, minibatch, zeroshot_opt, valset
        )

        if self.auto:
            self._print_auto_run_settings(num_trials, minibatch, valset)

        if minibatch and minibatch_size > len(valset):
            raise ValueError(
                f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}."
            )

        # Estimate LM calls and get user confirmation
        if requires_permission_to_run:
            if not self._get_user_confirmation(
                student,
                num_trials,
                minibatch,
                minibatch_size,
                minibatch_full_eval_steps,
                valset,
                program_aware_proposer,
            ):
                logger.info("Compilation aborted by the user.")
                return student  # Return the original student program

        # Initialize program and evaluator
        program = student.deepcopy()
        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=self.max_errors,
            display_table=False,
            display_progress=True,
        )

        # Step 1: Bootstrap few-shot examples
        # TODO: Uncomment later
        # demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed, teacher)
        demo_candidates = None

        # TODO: Need to combine the propose instructions and the optimize prompt parameters steps.
        # start off by moving the instruction proposal function into the optimize prompt parameters function.
        # then, simplify the prompt to allow only a single categorical variable to included or not. 
        # easiest is probably optimize overs tips in zero-shot setting. 

        # Step 2: Propose instruction candidates
        instruction_candidates = self._propose_instructions(
            program,
            trainset,
            demo_candidates,
            view_data_batch_size,
            program_aware_proposer,
            data_aware_proposer,
            tip_aware_proposer,
            fewshot_aware_proposer,
        )

        # If zero-shot, discard demos
        if zeroshot_opt:
            demo_candidates = None

        # Step 3: Find optimal prompt parameters
        best_program = self._optimize_prompt_parameters(
            program,
            instruction_candidates,
            demo_candidates,
            evaluate,
            valset,
            num_trials,
            minibatch,
            minibatch_size,
            minibatch_full_eval_steps,
            seed,
        )

        return best_program