??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.22v2.6.0-418-ga56365062368??
?
convolutional/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameconvolutional/kernel
?
(convolutional/kernel/Read/ReadVariableOpReadVariableOpconvolutional/kernel*&
_output_shapes
:
*
dtype0
|
convolutional/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconvolutional/bias
u
&convolutional/bias/Read/ReadVariableOpReadVariableOpconvolutional/bias*
_output_shapes
:*
dtype0
?
fully_connected/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_namefully_connected/kernel
?
*fully_connected/kernel/Read/ReadVariableOpReadVariableOpfully_connected/kernel*
_output_shapes
:	?*
dtype0
?
fully_connected/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namefully_connected/bias
y
(fully_connected/bias/Read/ReadVariableOpReadVariableOpfully_connected/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
		keras_api


signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api

0
1
2
3

0
1
2
3
 
?
#layer_metrics
$layer_regularization_losses
%metrics

&layers
	variables
'non_trainable_variables
trainable_variables
regularization_losses
 
 
 
 
?
(layer_metrics
)layer_regularization_losses
*metrics

+layers
	variables
,non_trainable_variables
trainable_variables
regularization_losses
`^
VARIABLE_VALUEconvolutional/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconvolutional/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
-layer_metrics
.layer_regularization_losses
/metrics

0layers
	variables
1non_trainable_variables
trainable_variables
regularization_losses
 
 
 
?
2layer_metrics
3layer_regularization_losses
4metrics

5layers
	variables
6non_trainable_variables
trainable_variables
regularization_losses
 
 
 
?
7layer_metrics
8layer_regularization_losses
9metrics

:layers
	variables
;non_trainable_variables
trainable_variables
regularization_losses
b`
VARIABLE_VALUEfully_connected/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEfully_connected/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
<layer_metrics
=layer_regularization_losses
>metrics

?layers
	variables
@non_trainable_variables
 trainable_variables
!regularization_losses
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
j
serving_default_input_1Placeholder*
_output_shapes
:	?*
dtype0*
shape:	?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1convolutional/kernelconvolutional/biasfully_connected/kernelfully_connected/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference_signature_wrapper_353
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(convolutional/kernel/Read/ReadVariableOp&convolutional/bias/Read/ReadVariableOp*fully_connected/kernel/Read/ReadVariableOp(fully_connected/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference__traced_save_580
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconvolutional/kernelconvolutional/biasfully_connected/kernelfully_connected/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_restore_602??
?

?
H__inference_fully_connected_layer_call_and_return_conditional_losses_536

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:2	
Softmaxc
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
F__inference_convolutional_layer_call_and_return_conditional_losses_143

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2	
BiasAddW
ReluReluBiasAdd:output:0*
T0*&
_output_shapes
:2
Relul
IdentityIdentityRelu:activations:0^NoOp*
T0*&
_output_shapes
:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:1(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
?
?
+__inference_convolutional_layer_call_fn_487

inputs!
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_convolutional_layer_call_and_return_conditional_losses_1432
StatefulPartitionedCallz
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:1(: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
?
^
@__inference_dropout_layer_call_and_return_conditional_losses_154

inputs

identity_1Y
IdentityIdentityinputs*
T0*&
_output_shapes
:2

Identityh

Identity_1IdentityIdentity:output:0*
T0*&
_output_shapes
:2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
?.
?
__inference__wrapped_model_109
input_1P
6tiny_conv_convolutional_conv2d_readvariableop_resource:
E
7tiny_conv_convolutional_biasadd_readvariableop_resource:K
8tiny_conv_fully_connected_matmul_readvariableop_resource:	?G
9tiny_conv_fully_connected_biasadd_readvariableop_resource:
identity??.tiny_conv/convolutional/BiasAdd/ReadVariableOp?-tiny_conv/convolutional/Conv2D/ReadVariableOp?0tiny_conv/fully_connected/BiasAdd/ReadVariableOp?/tiny_conv/fully_connected/MatMul/ReadVariableOp?
tiny_conv/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  2
tiny_conv/reshape/Shape?
%tiny_conv/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%tiny_conv/reshape/strided_slice/stack?
'tiny_conv/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'tiny_conv/reshape/strided_slice/stack_1?
'tiny_conv/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'tiny_conv/reshape/strided_slice/stack_2?
tiny_conv/reshape/strided_sliceStridedSlice tiny_conv/reshape/Shape:output:0.tiny_conv/reshape/strided_slice/stack:output:00tiny_conv/reshape/strided_slice/stack_1:output:00tiny_conv/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
tiny_conv/reshape/strided_slice?
!tiny_conv/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :12#
!tiny_conv/reshape/Reshape/shape/1?
!tiny_conv/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :(2#
!tiny_conv/reshape/Reshape/shape/2?
!tiny_conv/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!tiny_conv/reshape/Reshape/shape/3?
tiny_conv/reshape/Reshape/shapePack(tiny_conv/reshape/strided_slice:output:0*tiny_conv/reshape/Reshape/shape/1:output:0*tiny_conv/reshape/Reshape/shape/2:output:0*tiny_conv/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
tiny_conv/reshape/Reshape/shape?
tiny_conv/reshape/ReshapeReshapeinput_1(tiny_conv/reshape/Reshape/shape:output:0*
T0*&
_output_shapes
:1(2
tiny_conv/reshape/Reshape?
-tiny_conv/convolutional/Conv2D/ReadVariableOpReadVariableOp6tiny_conv_convolutional_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02/
-tiny_conv/convolutional/Conv2D/ReadVariableOp?
tiny_conv/convolutional/Conv2DConv2D"tiny_conv/reshape/Reshape:output:05tiny_conv/convolutional/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
2 
tiny_conv/convolutional/Conv2D?
.tiny_conv/convolutional/BiasAdd/ReadVariableOpReadVariableOp7tiny_conv_convolutional_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.tiny_conv/convolutional/BiasAdd/ReadVariableOp?
tiny_conv/convolutional/BiasAddBiasAdd'tiny_conv/convolutional/Conv2D:output:06tiny_conv/convolutional/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
tiny_conv/convolutional/BiasAdd?
tiny_conv/convolutional/ReluRelu(tiny_conv/convolutional/BiasAdd:output:0*
T0*&
_output_shapes
:2
tiny_conv/convolutional/Relu?
tiny_conv/dropout/IdentityIdentity*tiny_conv/convolutional/Relu:activations:0*
T0*&
_output_shapes
:2
tiny_conv/dropout/Identity?
tiny_conv/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
tiny_conv/flatten/Const?
tiny_conv/flatten/ReshapeReshape#tiny_conv/dropout/Identity:output:0 tiny_conv/flatten/Const:output:0*
T0*
_output_shapes
:	?2
tiny_conv/flatten/Reshape?
/tiny_conv/fully_connected/MatMul/ReadVariableOpReadVariableOp8tiny_conv_fully_connected_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/tiny_conv/fully_connected/MatMul/ReadVariableOp?
 tiny_conv/fully_connected/MatMulMatMul"tiny_conv/flatten/Reshape:output:07tiny_conv/fully_connected/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2"
 tiny_conv/fully_connected/MatMul?
0tiny_conv/fully_connected/BiasAdd/ReadVariableOpReadVariableOp9tiny_conv_fully_connected_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0tiny_conv/fully_connected/BiasAdd/ReadVariableOp?
!tiny_conv/fully_connected/BiasAddBiasAdd*tiny_conv/fully_connected/MatMul:product:08tiny_conv/fully_connected/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!tiny_conv/fully_connected/BiasAdd?
!tiny_conv/fully_connected/SoftmaxSoftmax*tiny_conv/fully_connected/BiasAdd:output:0*
T0*
_output_shapes

:2#
!tiny_conv/fully_connected/Softmax}
IdentityIdentity+tiny_conv/fully_connected/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp/^tiny_conv/convolutional/BiasAdd/ReadVariableOp.^tiny_conv/convolutional/Conv2D/ReadVariableOp1^tiny_conv/fully_connected/BiasAdd/ReadVariableOp0^tiny_conv/fully_connected/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 2`
.tiny_conv/convolutional/BiasAdd/ReadVariableOp.tiny_conv/convolutional/BiasAdd/ReadVariableOp2^
-tiny_conv/convolutional/Conv2D/ReadVariableOp-tiny_conv/convolutional/Conv2D/ReadVariableOp2d
0tiny_conv/fully_connected/BiasAdd/ReadVariableOp0tiny_conv/fully_connected/BiasAdd/ReadVariableOp2b
/tiny_conv/fully_connected/MatMul/ReadVariableOp/tiny_conv/fully_connected/MatMul/ReadVariableOp:H D

_output_shapes
:	?
!
_user_specified_name	input_1
?
?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_280

inputs+
convolutional_267:

convolutional_269:&
fully_connected_274:	?!
fully_connected_276:
identity??%convolutional/StatefulPartitionedCall?dropout/StatefulPartitionedCall?'fully_connected/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_1302
reshape/PartitionedCall?
%convolutional/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0convolutional_267convolutional_269*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_convolutional_layer_call_and_return_conditional_losses_1432'
%convolutional/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall.convolutional/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2292!
dropout/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_1622
flatten/PartitionedCall?
'fully_connected/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fully_connected_274fully_connected_276*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_1752)
'fully_connected/StatefulPartitionedCall?
IdentityIdentity0fully_connected/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp&^convolutional/StatefulPartitionedCall ^dropout/StatefulPartitionedCall(^fully_connected/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 2N
%convolutional/StatefulPartitionedCall%convolutional/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2R
'fully_connected/StatefulPartitionedCall'fully_connected/StatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
A
%__inference_reshape_layer_call_fn_467

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_1302
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?0
?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_422

inputsF
,convolutional_conv2d_readvariableop_resource:
;
-convolutional_biasadd_readvariableop_resource:A
.fully_connected_matmul_readvariableop_resource:	?=
/fully_connected_biasadd_readvariableop_resource:
identity??$convolutional/BiasAdd/ReadVariableOp?#convolutional/Conv2D/ReadVariableOp?&fully_connected/BiasAdd/ReadVariableOp?%fully_connected/MatMul/ReadVariableOpo
reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :12
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :(2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*&
_output_shapes
:1(2
reshape/Reshape?
#convolutional/Conv2D/ReadVariableOpReadVariableOp,convolutional_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02%
#convolutional/Conv2D/ReadVariableOp?
convolutional/Conv2DConv2Dreshape/Reshape:output:0+convolutional/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
2
convolutional/Conv2D?
$convolutional/BiasAdd/ReadVariableOpReadVariableOp-convolutional_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$convolutional/BiasAdd/ReadVariableOp?
convolutional/BiasAddBiasAddconvolutional/Conv2D:output:0,convolutional/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
convolutional/BiasAdd?
convolutional/ReluReluconvolutional/BiasAdd:output:0*
T0*&
_output_shapes
:2
convolutional/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMul convolutional/Relu:activations:0dropout/dropout/Const:output:0*
T0*&
_output_shapes
:2
dropout/dropout/Mul?
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*&
_output_shapes
:*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*&
_output_shapes
:2
dropout/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapedropout/dropout/Mul_1:z:0flatten/Const:output:0*
T0*
_output_shapes
:	?2
flatten/Reshape?
%fully_connected/MatMul/ReadVariableOpReadVariableOp.fully_connected_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%fully_connected/MatMul/ReadVariableOp?
fully_connected/MatMulMatMulflatten/Reshape:output:0-fully_connected/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
fully_connected/MatMul?
&fully_connected/BiasAdd/ReadVariableOpReadVariableOp/fully_connected_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&fully_connected/BiasAdd/ReadVariableOp?
fully_connected/BiasAddBiasAdd fully_connected/MatMul:product:0.fully_connected/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
fully_connected/BiasAdd?
fully_connected/SoftmaxSoftmax fully_connected/BiasAdd:output:0*
T0*
_output_shapes

:2
fully_connected/Softmaxs
IdentityIdentity!fully_connected/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp%^convolutional/BiasAdd/ReadVariableOp$^convolutional/Conv2D/ReadVariableOp'^fully_connected/BiasAdd/ReadVariableOp&^fully_connected/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 2L
$convolutional/BiasAdd/ReadVariableOp$convolutional/BiasAdd/ReadVariableOp2J
#convolutional/Conv2D/ReadVariableOp#convolutional/Conv2D/ReadVariableOp2P
&fully_connected/BiasAdd/ReadVariableOp&fully_connected/BiasAdd/ReadVariableOp2N
%fully_connected/MatMul/ReadVariableOp%fully_connected/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
\
@__inference_reshape_layer_call_and_return_conditional_losses_130

inputs
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :(2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapen
ReshapeReshapeinputsReshape/shape:output:0*
T0*&
_output_shapes
:1(2	
Reshapec
IdentityIdentityReshape:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_182

inputs+
convolutional_144:

convolutional_146:&
fully_connected_176:	?!
fully_connected_178:
identity??%convolutional/StatefulPartitionedCall?'fully_connected/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_1302
reshape/PartitionedCall?
%convolutional/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0convolutional_144convolutional_146*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_convolutional_layer_call_and_return_conditional_losses_1432'
%convolutional/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall.convolutional/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_1542
dropout/PartitionedCall?
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_1622
flatten/PartitionedCall?
'fully_connected/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fully_connected_176fully_connected_178*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_1752)
'fully_connected/StatefulPartitionedCall?
IdentityIdentity0fully_connected/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp&^convolutional/StatefulPartitionedCall(^fully_connected/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 2N
%convolutional/StatefulPartitionedCall%convolutional/StatefulPartitionedCall2R
'fully_connected/StatefulPartitionedCall'fully_connected/StatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
'__inference_tiny_conv_layer_call_fn_448

inputs!
unknown:

	unknown_0:
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_tiny_conv_layer_call_and_return_conditional_losses_2802
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
_
@__inference_dropout_layer_call_and_return_conditional_losses_229

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constr
dropout/MulMulinputsdropout/Const:output:0*
T0*&
_output_shapes
:2
dropout/Mulw
dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*&
_output_shapes
:*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:2
dropout/GreaterEqual~
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:2
dropout/Casty
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*&
_output_shapes
:2
dropout/Mul_1d
IdentityIdentitydropout/Mul_1:z:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
?
A
%__inference_dropout_layer_call_fn_509

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_1542
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
?
?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_338
input_1+
convolutional_325:

convolutional_327:&
fully_connected_332:	?!
fully_connected_334:
identity??%convolutional/StatefulPartitionedCall?dropout/StatefulPartitionedCall?'fully_connected/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_1302
reshape/PartitionedCall?
%convolutional/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0convolutional_325convolutional_327*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_convolutional_layer_call_and_return_conditional_losses_1432'
%convolutional/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall.convolutional/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2292!
dropout/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_1622
flatten/PartitionedCall?
'fully_connected/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fully_connected_332fully_connected_334*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_1752)
'fully_connected/StatefulPartitionedCall?
IdentityIdentity0fully_connected/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp&^convolutional/StatefulPartitionedCall ^dropout/StatefulPartitionedCall(^fully_connected/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 2N
%convolutional/StatefulPartitionedCall%convolutional/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2R
'fully_connected/StatefulPartitionedCall'fully_connected/StatefulPartitionedCall:H D

_output_shapes
:	?
!
_user_specified_name	input_1
?
?
'__inference_tiny_conv_layer_call_fn_193
input_1!
unknown:

	unknown_0:
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_tiny_conv_layer_call_and_return_conditional_losses_1822
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	?
!
_user_specified_name	input_1
?
?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_321
input_1+
convolutional_308:

convolutional_310:&
fully_connected_315:	?!
fully_connected_317:
identity??%convolutional/StatefulPartitionedCall?'fully_connected/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_1302
reshape/PartitionedCall?
%convolutional/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0convolutional_308convolutional_310*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_convolutional_layer_call_and_return_conditional_losses_1432'
%convolutional/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall.convolutional/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_1542
dropout/PartitionedCall?
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_1622
flatten/PartitionedCall?
'fully_connected/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fully_connected_315fully_connected_317*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_1752)
'fully_connected/StatefulPartitionedCall?
IdentityIdentity0fully_connected/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp&^convolutional/StatefulPartitionedCall(^fully_connected/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 2N
%convolutional/StatefulPartitionedCall%convolutional/StatefulPartitionedCall2R
'fully_connected/StatefulPartitionedCall'fully_connected/StatefulPartitionedCall:H D

_output_shapes
:	?
!
_user_specified_name	input_1
?
?
'__inference_tiny_conv_layer_call_fn_304
input_1!
unknown:

	unknown_0:
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_tiny_conv_layer_call_and_return_conditional_losses_2802
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	?
!
_user_specified_name	input_1
?
\
@__inference_flatten_layer_call_and_return_conditional_losses_162

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Const_
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	?2	
Reshape\
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
?
?
'__inference_tiny_conv_layer_call_fn_435

inputs!
unknown:

	unknown_0:
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_tiny_conv_layer_call_and_return_conditional_losses_1822
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
_
@__inference_dropout_layer_call_and_return_conditional_losses_504

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constr
dropout/MulMulinputsdropout/Const:output:0*
T0*&
_output_shapes
:2
dropout/Mulw
dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*&
_output_shapes
:*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:2
dropout/GreaterEqual~
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:2
dropout/Casty
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*&
_output_shapes
:2
dropout/Mul_1d
IdentityIdentitydropout/Mul_1:z:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
?
A
%__inference_flatten_layer_call_fn_525

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_1622
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
?
?
__inference__traced_save_580
file_prefix3
/savev2_convolutional_kernel_read_readvariableop1
-savev2_convolutional_bias_read_readvariableop5
1savev2_fully_connected_kernel_read_readvariableop3
/savev2_fully_connected_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_convolutional_kernel_read_readvariableop-savev2_convolutional_bias_read_readvariableop1savev2_fully_connected_kernel_read_readvariableop/savev2_fully_connected_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*@
_input_shapes/
-: :
::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
?
__inference__traced_restore_602
file_prefix?
%assignvariableop_convolutional_kernel:
3
%assignvariableop_1_convolutional_bias:<
)assignvariableop_2_fully_connected_kernel:	?5
'assignvariableop_3_fully_connected_bias:

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp%assignvariableop_convolutional_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp%assignvariableop_1_convolutional_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp)assignvariableop_2_fully_connected_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp'assignvariableop_3_fully_connected_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4c

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_5?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
H__inference_fully_connected_layer_call_and_return_conditional_losses_175

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:2	
Softmaxc
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
\
@__inference_reshape_layer_call_and_return_conditional_losses_462

inputs
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :(2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapen
ReshapeReshapeinputsReshape/shape:output:0*
T0*&
_output_shapes
:1(2	
Reshapec
IdentityIdentityReshape:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
^
@__inference_dropout_layer_call_and_return_conditional_losses_492

inputs

identity_1Y
IdentityIdentityinputs*
T0*&
_output_shapes
:2

Identityh

Identity_1IdentityIdentity:output:0*
T0*&
_output_shapes
:2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
?
?
!__inference_signature_wrapper_353
input_1!
unknown:

	unknown_0:
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_1092
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	?
!
_user_specified_name	input_1
?
\
@__inference_flatten_layer_call_and_return_conditional_losses_520

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Const_
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	?2	
Reshape\
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
?'
?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_384

inputsF
,convolutional_conv2d_readvariableop_resource:
;
-convolutional_biasadd_readvariableop_resource:A
.fully_connected_matmul_readvariableop_resource:	?=
/fully_connected_biasadd_readvariableop_resource:
identity??$convolutional/BiasAdd/ReadVariableOp?#convolutional/Conv2D/ReadVariableOp?&fully_connected/BiasAdd/ReadVariableOp?%fully_connected/MatMul/ReadVariableOpo
reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :12
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :(2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*&
_output_shapes
:1(2
reshape/Reshape?
#convolutional/Conv2D/ReadVariableOpReadVariableOp,convolutional_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02%
#convolutional/Conv2D/ReadVariableOp?
convolutional/Conv2DConv2Dreshape/Reshape:output:0+convolutional/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
2
convolutional/Conv2D?
$convolutional/BiasAdd/ReadVariableOpReadVariableOp-convolutional_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$convolutional/BiasAdd/ReadVariableOp?
convolutional/BiasAddBiasAddconvolutional/Conv2D:output:0,convolutional/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
convolutional/BiasAdd?
convolutional/ReluReluconvolutional/BiasAdd:output:0*
T0*&
_output_shapes
:2
convolutional/Relu?
dropout/IdentityIdentity convolutional/Relu:activations:0*
T0*&
_output_shapes
:2
dropout/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapedropout/Identity:output:0flatten/Const:output:0*
T0*
_output_shapes
:	?2
flatten/Reshape?
%fully_connected/MatMul/ReadVariableOpReadVariableOp.fully_connected_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%fully_connected/MatMul/ReadVariableOp?
fully_connected/MatMulMatMulflatten/Reshape:output:0-fully_connected/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
fully_connected/MatMul?
&fully_connected/BiasAdd/ReadVariableOpReadVariableOp/fully_connected_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&fully_connected/BiasAdd/ReadVariableOp?
fully_connected/BiasAddBiasAdd fully_connected/MatMul:product:0.fully_connected/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
fully_connected/BiasAdd?
fully_connected/SoftmaxSoftmax fully_connected/BiasAdd:output:0*
T0*
_output_shapes

:2
fully_connected/Softmaxs
IdentityIdentity!fully_connected/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp%^convolutional/BiasAdd/ReadVariableOp$^convolutional/Conv2D/ReadVariableOp'^fully_connected/BiasAdd/ReadVariableOp&^fully_connected/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:	?: : : : 2L
$convolutional/BiasAdd/ReadVariableOp$convolutional/BiasAdd/ReadVariableOp2J
#convolutional/Conv2D/ReadVariableOp#convolutional/Conv2D/ReadVariableOp2P
&fully_connected/BiasAdd/ReadVariableOp&fully_connected/BiasAdd/ReadVariableOp2N
%fully_connected/MatMul/ReadVariableOp%fully_connected/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
-__inference_fully_connected_layer_call_fn_545

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_1752
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
F__inference_convolutional_layer_call_and_return_conditional_losses_478

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2	
BiasAddW
ReluReluBiasAdd:output:0*
T0*&
_output_shapes
:2
Relul
IdentityIdentityRelu:activations:0^NoOp*
T0*&
_output_shapes
:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:1(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
?
^
%__inference_dropout_layer_call_fn_514

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2292
StatefulPartitionedCallz
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
3
input_1(
serving_default_input_1:0	?:
fully_connected'
StatefulPartitionedCall:0tensorflow/serving/predict:?[
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
		keras_api


signatures
A_default_save_signature
*B&call_and_return_all_conditional_losses
C__call__"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
*D&call_and_return_all_conditional_losses
E__call__"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*F&call_and_return_all_conditional_losses
G__call__"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"
_tf_keras_layer
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
*L&call_and_return_all_conditional_losses
M__call__"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
#layer_metrics
$layer_regularization_losses
%metrics

&layers
	variables
'non_trainable_variables
trainable_variables
regularization_losses
C__call__
A_default_save_signature
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
,
Nserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
(layer_metrics
)layer_regularization_losses
*metrics

+layers
	variables
,non_trainable_variables
trainable_variables
regularization_losses
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
.:,
2convolutional/kernel
 :2convolutional/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-layer_metrics
.layer_regularization_losses
/metrics

0layers
	variables
1non_trainable_variables
trainable_variables
regularization_losses
G__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
2layer_metrics
3layer_regularization_losses
4metrics

5layers
	variables
6non_trainable_variables
trainable_variables
regularization_losses
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7layer_metrics
8layer_regularization_losses
9metrics

:layers
	variables
;non_trainable_variables
trainable_variables
regularization_losses
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
):'	?2fully_connected/kernel
": 2fully_connected/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<layer_metrics
=layer_regularization_losses
>metrics

?layers
	variables
@non_trainable_variables
 trainable_variables
!regularization_losses
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?B?
__inference__wrapped_model_109input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_384
B__inference_tiny_conv_layer_call_and_return_conditional_losses_422
B__inference_tiny_conv_layer_call_and_return_conditional_losses_321
B__inference_tiny_conv_layer_call_and_return_conditional_losses_338?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_tiny_conv_layer_call_fn_193
'__inference_tiny_conv_layer_call_fn_435
'__inference_tiny_conv_layer_call_fn_448
'__inference_tiny_conv_layer_call_fn_304?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_reshape_layer_call_and_return_conditional_losses_462?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_reshape_layer_call_fn_467?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_convolutional_layer_call_and_return_conditional_losses_478?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_convolutional_layer_call_fn_487?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dropout_layer_call_and_return_conditional_losses_492
@__inference_dropout_layer_call_and_return_conditional_losses_504?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_dropout_layer_call_fn_509
%__inference_dropout_layer_call_fn_514?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_flatten_layer_call_and_return_conditional_losses_520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_flatten_layer_call_fn_525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_fully_connected_layer_call_and_return_conditional_losses_536?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_fully_connected_layer_call_fn_545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
!__inference_signature_wrapper_353input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_109j(?%
?
?
input_1	?
? "8?5
3
fully_connected ?
fully_connected?
F__inference_convolutional_layer_call_and_return_conditional_losses_478Z.?+
$?!
?
inputs1(
? "$?!
?
0
? |
+__inference_convolutional_layer_call_fn_487M.?+
$?!
?
inputs1(
? "??
@__inference_dropout_layer_call_and_return_conditional_losses_492Z2?/
(?%
?
inputs
p 
? "$?!
?
0
? ?
@__inference_dropout_layer_call_and_return_conditional_losses_504Z2?/
(?%
?
inputs
p
? "$?!
?
0
? v
%__inference_dropout_layer_call_fn_509M2?/
(?%
?
inputs
p 
? "?v
%__inference_dropout_layer_call_fn_514M2?/
(?%
?
inputs
p
? "??
@__inference_flatten_layer_call_and_return_conditional_losses_520O.?+
$?!
?
inputs
? "?
?
0	?
? k
%__inference_flatten_layer_call_fn_525B.?+
$?!
?
inputs
? "?	??
H__inference_fully_connected_layer_call_and_return_conditional_losses_536K'?$
?
?
inputs	?
? "?
?
0
? o
-__inference_fully_connected_layer_call_fn_545>'?$
?
?
inputs	?
? "??
@__inference_reshape_layer_call_and_return_conditional_losses_462O'?$
?
?
inputs	?
? "$?!
?
01(
? k
%__inference_reshape_layer_call_fn_467B'?$
?
?
inputs	?
? "?1(?
!__inference_signature_wrapper_353u3?0
? 
)?&
$
input_1?
input_1	?"8?5
3
fully_connected ?
fully_connected?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_321V0?-
&?#
?
input_1	?
p 

 
? "?
?
0
? ?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_338V0?-
&?#
?
input_1	?
p

 
? "?
?
0
? ?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_384U/?,
%?"
?
inputs	?
p 

 
? "?
?
0
? ?
B__inference_tiny_conv_layer_call_and_return_conditional_losses_422U/?,
%?"
?
inputs	?
p

 
? "?
?
0
? t
'__inference_tiny_conv_layer_call_fn_193I0?-
&?#
?
input_1	?
p 

 
? "?t
'__inference_tiny_conv_layer_call_fn_304I0?-
&?#
?
input_1	?
p

 
? "?s
'__inference_tiny_conv_layer_call_fn_435H/?,
%?"
?
inputs	?
p 

 
? "?s
'__inference_tiny_conv_layer_call_fn_448H/?,
%?"
?
inputs	?
p

 
? "?