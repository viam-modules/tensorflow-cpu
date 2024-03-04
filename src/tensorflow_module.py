from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional, Union
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

import tensorflow as tf


LOGGER = getLogger(__name__)


class TensorflowModule(MLModel, Reconfigurable):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "mlmodel"), "tensorflow")
     
    def __init__(self, name: str):
        super().__init__(name=name)
        
    @classmethod
    def new_service(cls,
                 config: ServiceConfig,
                 dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service
    
                      
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        model_dir = config.attributes.fields["model_dir"].string_value
        if model_dir == "":
            raise Exception(
                "please include the location of the Tensorflow SavedModel directory")
        return []

    def reconfigure(self,
            config: ServiceConfig,
            dependencies: Mapping[ResourceName, ResourceBase]):
        
        self.model_dir = config.attributes.fields["model_dir"].string_value
        # TODO: Validate that this is a valid model directory
        # Valid = it is a directory with a saved_model.pb file in it (?)

        # This is where we're gonna do the actual loading of the model
        self.model = tf.saved_model.load(self.model_dir)

        self.inputInfo = []
        self.outputInfo = []

        # Fill in the inputInfo as a list of tuples, each being a tensor with
        # (name, shape, underlying type, input index)
        f = self.model.signatures['serving_default']
        if len(f._arg_keywords) <= len(f.inputs):  # should always be true tbh
            for i in range(len(f._arg_keywords)):
                ff = f.inputs[i]
                if ff.dtype != "resource": # probably unneccessary to check now
                    info = (f._arg_keywords[i], self.shapeToList(ff.get_shape()),ff.dtype,i) 
                    self.inputInfo.append(info)
                    
        
    async def infer(self, input_tensors: Dict[str, NDArray], *, timeout: Optional[float]) -> Dict[str, NDArray]:
        """Take an already ordered input tensor as an array, make an inference on the model, and return an output tensor map.

        Args:
            input_tensors (Dict[str, NDArray]): A dictionary of input flat tensors as specified in the metadata

        Returns:
            Dict[str, NDArray]: A dictionary of output flat tensors as specified in the metadata
        """
        res = dict()

        #TODO: Make something to guess at the order the tensors are meant to be in (if more than 1)??
        # ["name0", "name1", "name3", "name2"] means the tensor "name3" in the map should be in spot 2 when fed into model (0-indexed)
        # Currently assuming the order they came in is the order they should go in 
        # ^^ Kinda valid since python dicts are ORDERED (since 3.7... and who ain't got at least 3.7?)
        
    
        inputVars = list(input_tensors.keys())[0]
        if len(inputVars) > len(self.inputInfo):
            raise Exception("there is fuckery about... too many inputs in map") #for now lol


        toInput = []
        for i in range(len(inputVars)):
            input = input_tensors[inputVars[i]]    
            input = tf.convert_to_tensor(input, dtype=self.inputInfo[i][2])
            input = tf.reshape(input, self.inputInfo[i][1])
            toInput.append(input)


        res = self.model(input)

        return res

    async def metadata(self, *, timeout: Optional[float]) -> Metadata:
        """Get the metadata (such as name, type, expected tensor/array shape, inputs, and outputs) associated with the ML model.

        Returns:
            Metadata: The metadata
        """

        inputInfo = []
        outputInfo = []
        for input in self.inputInfo:
            info = TensorInfo(name=input[0], shape=input[1] , data_type=input[2])
            inputInfo.append(info)

        for output in self.outputInfo:
            info = TensorInfo(name=output[0], shape=output[1] , data_type=output[2])
            outputInfo.append(info)

        return Metadata(name = "yada yada",
                        type = "blah blah",
                        input_info=inputInfo, 
                        output_info=outputInfo)
    
    async def do_command(self,
                        command: Mapping[str, ValueTypes],
                        *,
                        timeout: Optional[float] = None,
                        **kwargs):
        raise NotImplementedError
    
    def shapeToList(tensorShape) :
        out = tensorShape.as_list()
        for i in range(len(out)):
            if out[i]==None:
                out[i] = -1
        return out 