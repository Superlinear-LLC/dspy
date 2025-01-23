from langfuse import Langfuse
from typing import Optional, Any, Dict
import uuid
from datetime import datetime
from contextvars import ContextVar
from dspy.utils.callback import BaseCallback
from dspy.utils.callback import ACTIVE_CALL_ID

class LangfuseCallback(BaseCallback):
    """A callback that logs DSPy events to Langfuse."""
    
    def __init__(
        self,
        secret_key: str,
        public_key: str,
        host: str,
    ):
        """Initialize Langfuse callback.
        
        Args:
            secret_key: Langfuse secret key
            public_key: Langfuse public key
            host: Langfuse host URL 
        """
        super().__init__()
        
        # Initialize Langfuse client
        self.langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
        self.session_id = f"dspy-{uuid.uuid4()}"
        self.traces = {}
        self.observations = {}

    def _get_observation_status(self, exception: Optional[Exception]) -> tuple[str, Optional[str]]:
        """Get the level and status message for an observation."""
        level = "DEFAULT"
        status_message = None
        if exception:
            level = "ERROR"
            status_message = str(exception)
        return level, status_message

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        """A handler triggered when forward() method of a module (subclass of dspy.Module) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Module instance.
            inputs: The inputs to the module's forward() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        parent_call_id = ACTIVE_CALL_ID.get()
        parent_trace = self.traces.get(parent_call_id)
        parent_observation = self.observations.get(parent_call_id)

        if parent_trace is None and parent_observation is None: 
            trace = self.langfuse.trace(
                name=f"{instance.__class__.__name__}",
                id=call_id,  
                session_id=self.session_id,
                input=inputs,
            )
            self.traces[call_id] = trace
            return
        
        parent = parent_observation or parent_trace
        span = parent.span(
            name=f"{instance.__class__.__name__}",
            id=f"span-{call_id}",
            input=inputs,
        )
        self.observations[call_id] = span


    def on_module_end(
        self,
        call_id: str,
        outputs: Optional[Any],
        exception: Optional[Exception] = None,
    ):
        """A handler triggered after forward() method of a module (subclass of dspy.Module) is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the module's forward() method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        level, status_message = self._get_observation_status(exception)

        if hasattr(outputs, "_store"):
            output = outputs._store
        else:
            output = outputs

        if call_id in self.observations:
            # Nested module is done 
            observation = self.observations.pop(call_id)
            observation.end(
                output=output,
                level=level,
                status_message=status_message,
            )
        else:
            # Top-level module is done
            trace = self.traces.pop(call_id)
            trace.update(
                output=output,
            )



    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        """A handler triggered when __call__ method of dspy.LM instance is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The LM instance.
            inputs: The inputs to the LM's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        parent_call_id = ACTIVE_CALL_ID.get()
        parent_trace = self.traces.get(parent_call_id)
        parent_observation = self.observations.get(parent_call_id)

        if parent_trace is None and parent_observation is None: 
            trace = self.langfuse.trace(
                name=f"{instance.__class__.__name__}",
                id=call_id,  
                session_id=self.session_id,
                input=inputs,
            )
            self.traces[call_id] = trace
            parent_trace = trace
        
        parent = parent_observation or parent_trace
        generation = parent.generation(
            name=f"{instance.__class__.__name__}", 
            id=f"generation-{call_id}",
            input=inputs,
            model=instance.model,
        )
        self.observations[call_id] = generation

    def on_lm_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        """A handler triggered after __call__ method of dspy.LM instance is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the LM's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        level, status_message = self._get_observation_status(exception)
        if call_id in self.traces:
            trace = self.traces.pop(call_id)
            trace.update(
                output=outputs,
            )

        observation = self.observations.pop(call_id)
        observation.end(
            output=outputs,
            level=level,
            status_message=status_message,
        )

