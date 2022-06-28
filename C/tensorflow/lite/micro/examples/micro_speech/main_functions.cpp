/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/micro_speech/main_functions.h"

#include "audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/command_responder.h"
#include "tensorflow/lite/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/model.h"
#include "tensorflow/lite/micro/examples/micro_speech/recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "gpio.h"
#include "cy_pdl.h"
#include "cyhal.h"
#include "cybsp.h"

#ifdef TEST_MODE_HIL
// Include the necessary libraries for doing hardware in the loop testing with semihosting.
#include <stdio.h>
#include <stdlib.h>

// Populate the outdir which is initialized in main.c via argsparser to the main_functions.
extern "C" char* outdir;
// Create a file pointer for a log file with all information for testing the application
FILE* log_f;

#else 
// If semihosting and hardware in the loop testing is disabled the UART is the selected 
// as output for printf. For this we need the cy_retarget_io.h. 
#include "cy_retarget_io.h"
#endif

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *model_input = nullptr;
  FeatureProvider *feature_provider = nullptr;
  RecognizeCommands *recognizer = nullptr;
  int32_t previous_time = 0;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 10 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  int8_t feature_buffer[kFeatureElementCount];
  int8_t *model_input_buffer = nullptr;
} // namespace


cyhal_pwm_t pwm_obj;

// The name of this function is important for Arduino compatibility.
void setup()
{
  // Initalize the gpios so that we can interact with the outer world without Host PC 
  // for example with the User LED on the PSoC6 board. 
  gpio_init();

  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddConv2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddRelu() != kTfLiteOk)
  {
    return;
  }
  

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8))
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
  	
	/* Initialize retarget-io to use the debug UART port */
#ifndef TEST_MODE_HIL
	cy_retarget_io_init(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
	CY_RETARGET_IO_BAUDRATE);
#endif
  printf("Test UART printf\r\n");

	if (CY_RSLT_SUCCESS != cyhal_pwm_init(&pwm_obj, CYBSP_LED4, NULL)) {
			printf("PWM initilization failed!\n\r");
		}

	/* Enable global interrupts */
	__enable_irq();

  if (CY_RSLT_SUCCESS
      != cyhal_pwm_set_duty_cycle(&pwm_obj, GET_DUTY_CYCLE(50),
      PWM_LED_FREQ_HZ)) {
    printf("PWM failed to set dutycycle!\n\r");
  }

  if (CY_RSLT_SUCCESS != cyhal_pwm_start(&pwm_obj)) {
    printf("PWM failed to start!\n\r");
  }
#ifdef TEST_MODE_HIL
  log_f = fopen(outdir, "w");
  if (log_f == NULL)
  {
    printf("Opening outfile failed!\n");
  }
  fprintf(log_f, "TEST\n");
  printf("Logfile opened\n");
#endif

}

int32_t current_time = 0;
// The name of this function is important for Arduino compatibility.
void loop()
{
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp_ms();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }

#ifdef TEST_MODE_HIL
  fprintf(log_f, "previous_time %ld\n", previous_time);
  fprintf(log_f, "current_timestamp %ld\n", current_time);
  fprintf(log_f, "\nHow many new slices: %d\n", how_many_new_slices);
  printf("How many new slices: %d\n", how_many_new_slices);
#endif

  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0)
  {
#ifdef TEST_MODE_HIL
    exit(0);
#endif
    return;
  }
  
  // printf("audio_samples_size\r\n");
  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++)
  {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  gpio_0_set();
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  gpio_0_clear();
  // Obtain a pointer to the output tensor
  TfLiteTensor *output = interpreter->output(0);
  int8_t scores[4] = {0};
  scores[0] = output->data.int8[0];
  scores[1] = output->data.int8[1];
  scores[2] = output->data.int8[2];
  scores[3] = output->data.int8[3];
#ifdef TEST_MODE_HIL
  fprintf(log_f, "\nInference output: %d, %d, %d, %d \r\n", scores[0], scores[1], scores[2], scores[3]);
#endif

  // Determine whether a command was recognized based on the output of inference
  const char *found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;

  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  RespondToCommand(error_reporter, current_time, found_command, score,
                   is_new_command);

}
