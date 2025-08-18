import os
from typing import ClassVar, Mapping, Sequence, Dict, Optional, Tuple
from numpy.typing import NDArray
from typing_extensions import Self
from viam.services.mlmodel import MLModel, Metadata, TensorInfo
from viam.module.types import Reconfigurable
from viam.resource.types import Model, ModelFamily
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.utils import ValueTypes
from viam.logging import getLogger

import numpy as np
import google.protobuf.struct_pb2 as pb
import tensorflow as tf


LOGGER = getLogger(__name__)


class TensorflowModule(MLModel, Reconfigurable):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "mlmodel"), "tensorflow-cpu")

    def __init__(self, name: str):
        super().__init__(name=name)

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate_config(
        cls, config: ServiceConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        model_path_err = (
            "model_path must be the location of the Tensorflow SavedModel directory "
            "or the location of a Keras model file (.keras)"
        )

        model_path = config.attributes.fields["model_path"].string_value
        if model_path == "":
            raise Exception(model_path_err)

        # If it's a Keras model file, okay. Otherwise, it must be a SavedModel directory
        _, ext = os.path.splitext(model_path)
        if ext.lower() == ".keras":
            LOGGER.info(
                "Detected Keras model file at "
                + model_path
                + ". Please note Keras support is limited."
            )
            return ([], [])

        # Add trailing / if not there
        if model_path[-1] != "/":
            model_path = model_path + "/"

        # Check that model_path points to a dir with a pb file in it
        # and that the model file isn't too big (>500 MB)
        isValidSavedModel = False
        if not os.path.isdir(model_path):
            raise Exception(model_path_err)
        for file in os.listdir(model_path):
            if ".pb" in file:
                isValidSavedModel = True
                sizeMB = os.stat(model_path + file).st_size / (1024 * 1024)
                if sizeMB > 500:
                    LOGGER.warn(
                        "model file may be large for certain hardware ("
                        + str(sizeMB)
                        + "MB)"
                    )

        if not isValidSavedModel:
            raise Exception(model_path_err)

        return ([], [])

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        self.model_path = config.attributes.fields["model_path"].string_value
        self.label_path = config.attributes.fields["label_path"].string_value
        self.is_keras = False
        self.input_info = []  # input and output info are lists of tuples (name, shape, underlying type)
        self.output_info = []

        _, ext = os.path.splitext(self.model_path)
        if ext.lower() == ".keras":
            # If it's a Keras model, load it using the Keras API
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_keras = True

            # For now, we use first and last layer to get input and output info
            in_config = self.model.layers[0].get_config()
            out_config = self.model.layers[-1].get_config()

            # Keras model's output config's dtype is (sometimes?) a whole dict
            outType = out_config.get("dtype")
            if not isinstance(outType, str):
                outType = None

            self.input_info.append(
                (
                    in_config.get("name"),
                    in_config.get("batch_shape"),
                    in_config.get("dtype"),
                )
            )
            self.output_info.append(
                (out_config.get("name"), out_config.get("batch_shape"), outType)
            )

            return

        # This is where we do the actual loading of the SavedModel
        self.model = tf.saved_model.load(self.model_path)
        f = self.model.signatures["serving_default"]

        # f.inputs may include "empty" inputs as resources, but _arg_keywords only contains input tensor names
        if len(f._arg_keywords) <= len(f.inputs):  # should always be true tbh
            for i in range(len(f._arg_keywords)):
                ff = f.inputs[i]
                if ff.dtype != "resource":  # probably unneccessary to check now
                    info = (f._arg_keywords[i], prepShape(ff.get_shape()), ff.dtype)
                    self.input_info.append(info)

        for out in f.outputs:
            info = (out.name, prepShape(out.get_shape()), out.dtype)
            self.output_info.append(info)

    def _prepare_inputs(self, input_tensors: Dict[str, NDArray]) -> NDArray:
        """Prepare input tensors for inference."""
        input_vars = list(input_tensors.keys())
        if len(input_vars) > len(self.input_info):
            raise Exception(
                f"there are more input tensors ({len(input_vars)}) than the model expected ({len(self.input_info)})"
            )


        input_list = []
        for i, var_name in enumerate(input_vars):
            input_t = input_tensors[var_name]
            input_t = tf.convert_to_tensor(input_t, dtype=self.input_info[i][2])
            input_list.append(input_t)


        if len(input_vars) == 1:
            return np.squeeze(np.asarray(input_list), axis=0)
        return np.asarray(input_list)

    def _run_inference(self, data: NDArray):
        """Run inference on the model."""
        try:
            return self.model(data)
        except (ValueError, TypeError) as direct_err:
            try:
                f = self.model.signatures["serving_default"]
                _, kwargs_spec = f.structured_input_signature
                input_name = next(iter(kwargs_spec.keys()))
                return f(**{input_name: data})
            except (ValueError, TypeError) as serving_err:
                raise Exception(
                    f"both direct model inference and serving_default failed: direct_err={direct_err}; serving_default_err={serving_err}"
                ) from serving_err

    def _process_outputs(self, res) -> Dict[str, NDArray]:
        """Process model outputs into the expected format."""
        if len(self.output_info) < len(res):
            raise Exception(
                f"there are more output tensors ({len(res)}) than the model expected ({len(self.output_info)})"
            )

        out = {}
        if isinstance(res, dict):
            if len(res) == 1:
                out[self.output_info[0][0]] = np.asarray(next(iter(res.values())))
            else:
                for k, v in res.items():
                    out[k] = np.asarray(v)
        elif isinstance(res, tuple):
            for i, v in enumerate(res):
                out[f"output_{i}"] = np.asarray(v)
        elif len(res) == 1:
            out[self.output_info[0][0]] = np.asarray(res[0])
        else:
            raise Exception(f"Unexpected output type and length: {type(res)} and {len(res)}")
        return out

    async def infer(
        self,
        input_tensors: Dict[str, NDArray],
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, NDArray]:
        """Take an already ordered input tensor as an array, make an inference on the model, and return an output tensor map.

        Args:
            input_tensors (Dict[str, NDArray]): A dictionary of input flat tensors as specified in the metadata

        Returns:
            Dict[str, NDArray]: A dictionary of output flat tensors as specified in the metadata
        """
        data = self._prepare_inputs(input_tensors)

        if self.is_keras:
            res = self.model.predict(data, verbose=0)
            return {"output_0": np.asarray(res)}

        res = self._run_inference(data)
        return self._process_outputs(res)

    async def metadata(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Metadata:
        """Get the metadata (such as name, type, expected tensor/array shape, inputs, and outputs) associated with the ML model.

        Returns:
            Metadata: The metadata
        """

        extra = pb.Struct()
        extra["labels"] = self.label_path

        # Fill out input and output info
        input_info = []
        output_info = []
        for inputT in self.input_info:
            info = TensorInfo(
                name=inputT[0],
                shape=prepShape(inputT[1]),
                data_type=prepType(inputT[2], self.is_keras),
            )
            input_info.append(info)

        for output in self.output_info:
            info = TensorInfo(
                name=output[0],
                shape=prepShape(output[1]),
                data_type=prepType(output[2], self.is_keras),
                extra=extra,
            )
            output_info.append(info)

        return Metadata(
            name="tensorflow_model", input_info=input_info, output_info=output_info
        )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        return NotImplementedError


# Want to return a list of ints (-1 for None)
def prepShape(tensorShape):
    if tensorShape is None:
        return None
    out = []
    for t in list(tensorShape):
        if t is None:
            out.append(-1)
        else:
            out.append(t)
    return out


# Want to return a simple string ("float32", "int64", etc.)
def prepType(tensorType, is_keras):
    if tensorType is None:
        return "unknown"
    if hasattr(tensorType, 'name'):
        return tensorType.name
    if not isinstance(tensorType, str):
        return "unknown"
    if is_keras:
        return tensorType

    # The dtype for SavedModel uses an escaped apostrophe around the actual type name so use that
    s = str(tensorType)
    inds = [i for i, letter in enumerate(s) if letter == "'"]
    return s[inds[0] + 1 : inds[1]]
