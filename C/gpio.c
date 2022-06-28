/* ===========================================================================
** Copyright (C) 2012-2021 Infineon Technologies AG
** All rights reserved.
** ===========================================================================
**
** ===========================================================================
** This document contains proprietary information of Infineon Technologies AG. 
** Passing on and copying of this document, and communication of its contents 
** is not permitted without Infineon's prior written authorisation.
** ===========================================================================
*/
/**
 * @file    gpio.c
 * @brief   Implementation of PSoC gpio handling
 *
 * @note
 * Version: 
 *
 */

/*
==============================================================================
   1. INCLUDE FILES
==============================================================================
*/

#include "gpio.h"
#include "cybsp.h"
#include "cyhal.h"

/*
==============================================================================
   2. LOCAL DEFINITIONS
==============================================================================
*/

/*
==============================================================================
   3. LOCAL TYPES
==============================================================================
*/

/*
==============================================================================
   4. DATA
==============================================================================
*/

static bool gpio_is_initialized = false;

/*
==============================================================================
   5. LOCAL FUNCTION PROTOTYPES
==============================================================================
*/

/*
==============================================================================
  6. LOCAL FUNCTIONS
==============================================================================
*/

void gpio_init(void)
{
    {
        cy_rslt_t result;

        /* Initialize the device and board peripherals */
        result = cybsp_init();

        /* Board init failed. Stop program execution */
        if (result != CY_RSLT_SUCCESS)
        {
            CY_ASSERT(0);
        }
    }

    {
        cy_rslt_t result;

        // Set up LEDs as outputs
        // /* Initialize the User LED */
        // result = cyhal_gpio_init((cyhal_gpio_t) CYBSP_USER_LED,
        //         CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG,
        //         0);
        /* Initialize pin-toggling on P9_3 and P9_5 for use with logic analyzer */
        result = cyhal_gpio_init((cyhal_gpio_t) P9_3,
                CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG,
                0);
        result |= cyhal_gpio_init((cyhal_gpio_t) P9_5,
                CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG,
                0);
        result |= cyhal_gpio_init((cyhal_gpio_t) P9_0,
                CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG,
                0);
        result |= cyhal_gpio_init((cyhal_gpio_t) P9_2,
                CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG,
                0);
        result |= cyhal_gpio_init((cyhal_gpio_t) P9_1,
                CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG,
                0);
        

        /* retarget-io init failed. Stop program execution */
        if (result != CY_RSLT_SUCCESS)
        {
            CY_ASSERT(0);
        }

        // Ensure all pins are cleared
        cyhal_gpio_write((cyhal_gpio_t) CYBSP_USER_LED, CYBSP_LED_STATE_OFF);
        cyhal_gpio_write((cyhal_gpio_t) P9_3, 0);
        cyhal_gpio_write((cyhal_gpio_t) P9_5, 0);
        cyhal_gpio_write((cyhal_gpio_t) P9_0, 0);
        cyhal_gpio_write((cyhal_gpio_t) P9_2, 0);
        cyhal_gpio_write((cyhal_gpio_t) P9_1, 0);
    }

    gpio_is_initialized = true;
}

void gpio_0_set(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_0, 1);
    }
}

void gpio_0_clear(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_0, 0);
    }
}

void gpio_0_toggle(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_toggle((cyhal_gpio_t) P9_0);
    }
}

void gpio_1_set(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_5, 1);
    }
}

void gpio_1_clear(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_5, 0);
    }
}

void gpio_1_toggle(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_toggle((cyhal_gpio_t) P9_5);
    }
}

void gpio_2_set(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_3, 1);
    }
}

void gpio_2_clear(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_3, 0);
    }
}

void gpio_2_toggle(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_toggle((cyhal_gpio_t) P9_3);
    }
}

void gpio_3_set(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_2, 1);
    }
}

void gpio_3_clear(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_2, 0);
    }
}

void gpio_3_toggle(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_toggle((cyhal_gpio_t) P9_2);
    }
}

void gpio_4_set(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_1, 1);
    }
}

void gpio_4_clear(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_write((cyhal_gpio_t) P9_1, 0);
    }
}

void gpio_4_toggle(void)
{
    if (gpio_is_initialized == true)
    {
        cyhal_gpio_toggle((cyhal_gpio_t) P9_1);
    }
}
/*
==============================================================================
   7. EXPORTED FUNCTIONS
==============================================================================
*/

/* --- End of File ------------------------------------------------ */