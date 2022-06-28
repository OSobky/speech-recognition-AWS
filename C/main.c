/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "cyhal.h"
#include "cybsp.h"

#ifdef TEST_MODE_HIL
#include "stdbool.h"
#include "stdlib.h"
#include "stdio.h"

extern void initialise_monitor_handles(void);

char* outdir;
char* indir;


// arg[0] = outdir
void args_parse(int argc, char* argv[])
{

    printf("argc = %d\r\n", argc);
    for (uint16_t i=0; i<argc; i++)
    {
    	printf("argv[%d] = %s\r\n", i, argv[i]);
    }


    outdir = NULL;
    if (argc > 0)
    {
        outdir = argv[0];
    	printf("Using %s as output directory.\r\n", outdir);
    }

    indir = NULL;
    if (argc > 1)
    {
        indir = argv[1];
    	printf("Using %s as input directory.\r\n", indir);
    }

}
#else

#include "cy_retarget_io.h"

#endif

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char* argv[]) {
// printf("\x1b[2J\x1b[;H");
#ifdef TEST_MODE_HIL
  initialise_monitor_handles();
  args_parse(argc, argv);
#endif
  setup();
  for(;;) {
    loop();
  }
}