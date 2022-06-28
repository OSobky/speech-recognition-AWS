/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/examples/micro_speech/feature_provider.h"

#include "audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "cy_retarget_io.h"

#include "stdlib.h"

#ifdef TEST_MODE_HIL
extern FILE* log_f;
#endif

FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(
    tflite::ErrorReporter* error_reporter, int32_t last_time_in_ms,
    int32_t time_in_ms, int* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Requested feature_data_ size %d doesn't match %d",
                         feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }
  int slices_needed = 0;
  int current_step;

  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
#ifndef TEST_MODE_HIL
  const int last_step = (last_time_in_ms / kFeatureSliceStrideMs);
  current_step = (time_in_ms / kFeatureSliceStrideMs);

  slices_needed = current_step - last_step;
#endif
  // If this is the first call, make sure we don't use any cached information.

  if (is_first_run_) {
    TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    
#ifdef TEST_MODE_HIL
    RecordAudioStart(
		AUDIO_PROVIDER_BUTTON_NOT_USED,
		AUDIO_PROVIDER_TESTMODE_SEMIHOSTING_SOURCE
	  );

    // The testmode semihosting source reads when initalizing 16000 samples. 
    // so we need to reset the last step and current step and the last and time in ms 

    last_time_in_ms = 0;
    time_in_ms = 1000;

    const int last_step = (last_time_in_ms / kFeatureSliceStrideMs);
    current_step = (time_in_ms / kFeatureSliceStrideMs);

    slices_needed = current_step - last_step;

#else
    slices_needed = kFeatureSliceCount;
    RecordAudioStart(
      AUDIO_PROVIDER_BUTTON_NOT_USED,
      AUDIO_PROVIDER_TESTMODE_DISABLED
	  );
#endif
    is_first_run_ = false;
  }
  if (slices_needed > kFeatureSliceCount) {
    slices_needed = kFeatureSliceCount;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;
  // If we can avoid recalculating some slices, just move the existing data
  // up in the spectrogram, to perform something like this:
  // last time = 80ms          current time = 120ms
  // +-----------+             +-----------+
  // | data@20ms |         --> | data@60ms |
  // +-----------+       --    +-----------+
  // | data@40ms |     --  --> | data@80ms |
  // +-----------+   --  --    +-----------+
  // | data@60ms | --  --      |  <empty>  |
  // +-----------+   --        +-----------+
  // | data@80ms | --          |  <empty>  |
  // +-----------+             +-----------+
  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      int8_t* dest_slice_data =
          feature_data_ + (dest_slice * kFeatureSliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src_slice_data =
          feature_data_ + (src_slice * kFeatureSliceSize);
      for (int i = 0; i < kFeatureSliceSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount;
         ++new_slice) {
      const int new_step = (current_step - kFeatureSliceCount + 1) + new_slice;
#ifdef TEST_MODE_HIL
      fprintf(log_f, "\nnew_step: %d\n", new_step);
      fprintf(log_f, "current_step: %d\n", current_step);
#endif
      const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs) - kFeatureSliceDurationMs;
      int16_t audio_data[kMaxAudioSampleSize];
      uint32_t audio_data_num_samples;

      // TODO(petewarden): Fix bug that leads to non-zero slice_start_ms
      if (GetAudioData(
        (slice_start_ms > 0 ? slice_start_ms : 0),
        kFeatureSliceDuration_Ms,
        audio_data,
        &audio_data_num_samples) == false)
      {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Number of requested audio samples %d exceeds buffer size %d",
                             audio_data_num_samples, kAudioCaptureBufferSize);
        return kTfLiteError;
      }
      // GetAudioData() fills 480 samples into audio_data
      // but we pass the whole 512 sample buffer to GenerateMicroFeatures().
      // So, we pad zeros to guarantee that there is no artifact appended
      // to the actual audio data.
      memset(&audio_data[audio_data_num_samples], 0,
             (kMaxAudioSampleSize-audio_data_num_samples)*sizeof(audio_data[0]));

#ifdef TEST_MODE_HIL
      fprintf(log_f, "\nInput DATA:");
      for (int16_t i=0; i < 480; i++)
      {
        fprintf(log_f, "%d, ", audio_data[i]);
      }

      printf("Initialize Micro Features\n");

      TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
      if (init_status != kTfLiteOk) {
        return init_status;
      }
#endif

      int8_t* new_slice_data = feature_data_ + (new_slice * kFeatureSliceSize);
      size_t num_samples_read;
      TfLiteStatus generate_status = GenerateMicroFeatures(
        error_reporter, audio_data, 480, kFeatureSliceSize,
        new_slice_data, &num_samples_read);
      if (generate_status != kTfLiteOk) {
        return generate_status;
      }
#ifdef TEST_MODE_HIL
      fprintf(log_f, "\nOutput DATA: ");
      for (int16_t i=0;i<40; ++i)
      {
        fprintf(log_f, "%d, ", new_slice_data[i]);
      }
#endif
    }
  }
  return kTfLiteOk;
}
