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

#ifndef GPIO_H
#define GPIO_H
/**
 * @file    gpio.h
 * @brief   Interface declarations for the definition of the gpios.
 *
 * @note
 * Version: 
 *
 */

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

/*
==============================================================================
   1. INCLUDE FILES
==============================================================================
*/

/*
==============================================================================
   2. DEFINITIONS
==============================================================================
*/


/*
==============================================================================
   3. TYPES
==============================================================================
*/

/*
==============================================================================
   4. EXPORTED DATA
==============================================================================
*/

/*
==============================================================================
   5. FUNCTION PROTOTYPES
==============================================================================
*/

extern void gpio_init(void);

extern void gpio_0_set(void);
extern void gpio_0_clear(void);
extern void gpio_0_toggle(void);
extern void gpio_1_set(void);
extern void gpio_1_clear(void);
extern void gpio_1_toggle(void);
extern void gpio_2_set(void);
extern void gpio_2_clear(void);
extern void gpio_2_toggle(void);
extern void gpio_3_set(void);
extern void gpio_3_clear(void);
extern void gpio_3_toggle(void);
extern void gpio_4_set(void);
extern void gpio_4_clear(void);
extern void gpio_4_toggle(void);

/*
==============================================================================
   6. INLINE FUNCTIONS
==============================================================================
*/

#ifdef __cplusplus
}
#endif

#endif  // header guard

/* --- End of File ------------------------------------------------ */