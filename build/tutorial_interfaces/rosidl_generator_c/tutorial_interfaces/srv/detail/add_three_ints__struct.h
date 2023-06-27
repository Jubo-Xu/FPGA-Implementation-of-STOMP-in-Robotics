// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tutorial_interfaces:srv/AddThreeInts.idl
// generated code does not contain a copyright notice

#ifndef TUTORIAL_INTERFACES__SRV__DETAIL__ADD_THREE_INTS__STRUCT_H_
#define TUTORIAL_INTERFACES__SRV__DETAIL__ADD_THREE_INTS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/AddThreeInts in the package tutorial_interfaces.
typedef struct tutorial_interfaces__srv__AddThreeInts_Request
{
  int64_t a;
  int64_t b;
  int64_t c;
} tutorial_interfaces__srv__AddThreeInts_Request;

// Struct for a sequence of tutorial_interfaces__srv__AddThreeInts_Request.
typedef struct tutorial_interfaces__srv__AddThreeInts_Request__Sequence
{
  tutorial_interfaces__srv__AddThreeInts_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tutorial_interfaces__srv__AddThreeInts_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/AddThreeInts in the package tutorial_interfaces.
typedef struct tutorial_interfaces__srv__AddThreeInts_Response
{
  int64_t sum;
} tutorial_interfaces__srv__AddThreeInts_Response;

// Struct for a sequence of tutorial_interfaces__srv__AddThreeInts_Response.
typedef struct tutorial_interfaces__srv__AddThreeInts_Response__Sequence
{
  tutorial_interfaces__srv__AddThreeInts_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tutorial_interfaces__srv__AddThreeInts_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TUTORIAL_INTERFACES__SRV__DETAIL__ADD_THREE_INTS__STRUCT_H_
