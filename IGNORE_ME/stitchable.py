from typing import Type, Union, NoReturn, Callable, List, TypeAlias, Tuple
from warnings import warn

import torch
import torch.nn as nn

# A version of a model that is stitch-enabled. That is to say, you pass a model into the constructor
# and it converts it into a form that can output at any intermediate layer (terminating computation)
# and can take input at any intermediate layer without doing any prior computation.

# NOTE that this is primarily meant to enable stitching on pretrained ResNets from https://arxiv.org/pdf/1512.03385.pdf
# and related papers. In general, vision models may work, but NLP and Graph models are NOT supported. See the requirements
# below to understand why.

# Requirements:
# 1. Your model must be static. If you use torch.functional in your model (even for something as simple
#    as ReLU) this will NOT work. Instead, use modules from torch.nn. The functionality must NOT change
#    from run to run. No error code or message is provided if your model is dynamic, it just will have
#    different behavior.
# 2. From (1) we know that your model uses only modules. However, your model must also use exclusively
#    torch.nn modules as well as +, -, *, matrix multiplication, and other Pytorch primitives WITHOUT
#    BRANCHING. If you have any branching (i.e. based on data), Stitchable will, without an error code,
#    produce the wrong functionality. Make sure (i.e. using `print(model)`) to ensure that you are composing
#    exclusively Pytorch-native modules in a branchless fashion.
# 3. Your model must be an nn.Module since we need functionality to read its parameters and their relations.
# 4. The input of your model should be a tensor and so should the output.

# Interface:
# 1. Constructor `Stitchable(model: nn.Module) -> NoReturn`
#    that takes in a model that is an nn.Module following the requirements. Proper usage is to simply pass in your
#    model and then
#
# 2. Function `forward(x: torch.Tensor, into: Optional[Union[str, int]] = None, outfrom: Optional[Union[str, int]] = None)`
#    that enables you to declare where to inject your input into as well as where to vent your output from. By default
#    input should go into the first layer and come outfrom the last layer (i.e. if you pass in None).
# 3. Functions for sanity testing:
#    `sanityTest_compareAcc(original: nn.Module, verbose: bool = True) -> bool`
#    that will run sanity tests on your stitched model by comparing its accuracy with the original model (should be the same).


class Stitchable(nn.Module):
    def __init__(self, model: nn.Module) -> NoReturn:
        # TODO
        stateDict = model.state_dict()
        print(type(stateDict))
        print("\n".join(list(stateDict.keys())))
        print(model)
        raise NotImplementedError

    def forward(self, x: torch.Tensor,
                into: Optional[Union[str, int]] = None,
                outfrom: Optional[Union[str, int]] = None) -> torch.Tensor:
        return x  # TODO

    def sanityTest_compareAcc(self, original: nn.Module, verbose: bool = True) -> bool:
        return False

# A stitched model is the model that uses the first n (out of N > n) layers from the N-layer sending network and last M - m
# (out of M > m) layers from the M-layer recieving network. Between those two layers it inserts a stitch
# for which the user is expected to pass a function. Thus you can think of the output as
# out = reciever_m_M(stitch(sender_0_n(x))) if we use underscores to illustrate what sections of the networks
# are being used.

# Interface:
# 1. Constructor `Stitched(sender: Stitchable, reciever: Stitchable,
#     sendingLayer: Union[str, int], recievingLayer: Union[str, int],
#     create_stitch: Callable[[int, int], nn.Module]) -> NoReturn that stitches the first senderLayer layers of
#     the sender (network) through the stitch into the latter M - recievingLayer layers of the reciever network. You
#     must pass in a function called create_stitch which takes in two layers and returns the stitch. Use a closure
#     to encode any information regarding the two models' sizes and stitch type.
# 2. Function `forward(x: torch.Tensor) -> torch.Tensor` that will simply compute the output as described above.
# 3. Utility getters (for parameters you passed in):
# 3.1 get_stitch() -> nn.Module
# 3.2 get_sender() -> Stitchable
# 3.3 get_reciever() -> Stitchable
# 3.4 get_sendingLayer() -> Union[str, int]
# 3.5 get_recievingLayer() -> Union[str, int]
# 4. Utility Sanity Tests:
# 4.1 `sanityTest() -> bool` which will run all the non-static sanity tests below and make sure that your model is OK to train stitches
#     and that, importantly, you can trust these results. This is meant to be used with an assert statement and will issue warnings
#     before failure. The number of parameters should be the sum of the numbers of parameters in the models plus the stitch (due to
#     the way stitchable works).
# 4.2 A static sanity tester `sanityTest_ptrsShareOk(other: Stitched) -> bool` which will test that the pointers of your model
#  differ when you differ the parameter or the model.
# 4.3 `sanityTest_gradientUpdateOk() -> bool` which will test that if you do training your gradients update
#     properly (i.e. only the stitch should update, while the rest of the model, that is parameters from the sender and the reciever,
#     should NOT update).


class Stitched(nn.Module):
    def __init__(self,
                 sender: Stitchable, reciever: Stitchable,
                 sendingLayer: Union[str, int], recievingLayer: Union[str, int],
                 create_stitch: Callable[[int, int], nn.Module]) -> NoReturn:
        super(StitchedResnet, self).__init__()

        self.sender = sender
        self.reciever = reciever
        self.sndLayer = sendingLayer
        self.rcvLayer = recievingLayer

        self.stitch = create_stitch(
            self.out_layer,
            self.in_layer)

        # NOTE that we only support same model by reference
        self.sameModel = sender == reciever

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sendert(x, outfrom=self.sndLayer)
        out = self.stitch(out)
        out = self.reciever(x, into=self.rcvLayer)
        return out

    def get_stitch(self) -> nn.Module:
        return self.stitch

    def get_sender(self) -> Stitchable:
        return self.sender

    def get_reciever(self) -> Stitchable:
        return self.reciever

    def get_sendingLayer(self) -> Union[str, int]:
        return self.sendingLayer

    def get_recievingLayer(self) -> Union[str, int]:
        return self.recievingLayer

    def sanityTest(self) -> bool:
        return (
            self.sanityTest_rightNumberParams() and
            self.sanityTest_gradientUpdateOk())

    def sanityTest_rightNumberParams(self) -> bool:
        sane = True
        failStr = "sanityTest_rightNumberParams FAIL"

        stitch = self.get_stitch()
        sender = self.get_sender()
        reciever = self.get_reciever()

        numParams = len(list(self.parameters()))
        numStitchParams = len(list(stitch.parameters()))
        numSenderParams = len(list(sender.parameters()))
        numRecieverParams = len(list(reciever.parameters()))

        if numParams <= 0:
            warn(
                f"{failStr}: Stitched model has {numParams} parameters (should be more than zero)")
            sane = False
        if numStitchParams <= 0:
            warn(
                f"{failStr}: Stitch has {numStitchParams} parameters (should be more than zero)")
            sane = False
        if numSenderParams <= 0:
            warn(
                f"{failStr}: Model 1 has {numSenderParams} parameters (should be more than zero)")
            sane = False
        if numRecieverParams <= 0:
            warn(
                f"{failStr}: Model 2 has {numRecieverParams} parameters (should be more than zero)")
            sane = False

        total_params = numSenderParams + numStitchParams
        if self.sameModel and total_params != numParams:
            warn(f"{failStr}: Self-stitch, but number of stitched model parameters is {numParams}, when it should be {total_params} = (sender/reciever) {numSenderParams} + (stitch) {numStitchParams}")
            sane = False
        total_params += numRecieverParams
        if not self.sameModel and total_params != numParams:
            warn(f"{failStr}: Non-self-stitch, but number of stitched model parameters is {numParams}, when it should be {total_params} = (sender) {numSenderParams} + (stitch) {numStitchParams} + (reciever) {numRecieverParams}")
            sane = False

        return sane

    def sanityTest_gradientUpdateOk(self) -> bool:
        failStr = "sanityTest_gradientUpdateOk FAIL"
        # TODO move this to a utility file (potentially create an object to do this)

        def pclone(model): return [p.data.detach().clone()
                                   for p in model.parameters()]
        def listeq(l1, l2): return min((torch.eq(a, b).int().min().item()
                                        for a, b in zip(l1, l2))) == 1
        sndParams0, recvParams0, stitchParams0 = pclone(
            self.sender), pclone(self.reciever), pclone(self.stitch)

        # TODO create clean training modular code for this...
        # scale_train_model(
        #     self, optimizer, criterion, scaler,
        #     trainloader, None,
        #     epochs=1, print_every=2, save_every=2, save_name=None,
        #     tensorboardx_writer=None, tensorboardx_scalar=None)
        raise NotImplementedError

        sndParams1, recvParams1, stitchParams1 = pclone(
            sender), pclone(reciever), pclone(stitch)
        if not listeq(sender_params, sndParams1):
            warn(f"{failStr}: Model 1 was updated by stitch training")
            sane = False
        if not listeq(reciever_params, recvParams1):
            warn(f"{failStr}: Model 2 was updated by stitch training")
            sane = False

        if listeq(stitch_params, stitchParams1):
            warn(f"{failStr}: Model 3 was not updated by stitch training")
            sane = False
        return sane

    @staticmethod
    def sanityTest_ptrsShareOk(stitchedModels: List[List[Stitched]]) -> bool:
        failStr = "sanityTest_ptrsShareOk FAIL"

        layers = range(len(stitchedModels))

        sane = True
        for l1a in layers:
            for l2a in layers:
            model_a = stitchedModels[l1a][l2a]
            if not model_a is None:
                for l1b in layers:
                    for l2b in layers:
                        model_b = stitchedModels[l1b][l2b]
                        if not model_b is None:
                            model_a_st = model_a.get_stitch()
                            model_b_st = model_b.get_stitch()
                            sameStitch = l1a == l1b and l2a == l2b
                            if sameStitch:
                                if model_a != model_b:
                                    warn(
                                        f"{failStr}: Model at {l1a}{l2a} was not same as itself")
                                    sane = False
                                if model_a_st != model_b_st:
                                    warn(
                                        f"{failStr}: Model at {l1a}{l2a} had stitch that was not same as itself")
                                    sane = False
                            for p_a_idx, p_a in enumerate(model_a_st.parameters()):
                                for p_b_idx, p_b in enumerate(model_b_st.parameters()):
                                    ptr_a = p_a.data_ptr()
                                    ptr_b = p_b.data_ptr()
                                    same_param = sameStitch and p_a_idx == p_b_idx
                                    ptr_ok = same_param ^ (ptr_a != ptr_b)
                                    if not ptr_ok:
                                        err = "not" if sameParam else ""
                                        errinv = "" if sameParam else "not"
                                        warn(
                                            f"{failStr}: Stitches from layers({l1a}->{l2a})param({p_a_idx}) and layers({l1b}-{l2b})param({p_b_idx}):\n\tptrs are {err} equal but should {errinv} be\n\tptrs were {ptr_a} and {ptr_b}")
                                        sane = False
            return sane


# TODO
ValidStitchesDict: TypeAlias = Dict[Union[str, int], List[Union[str, int]]]
ModelsToStitch: TypeAlias = Dict[str, Tuple[nn.Module, ValidStitchesDict]]


class StitchExperiments(object):
    def __init__(self) -> NoReturn:
        pass
