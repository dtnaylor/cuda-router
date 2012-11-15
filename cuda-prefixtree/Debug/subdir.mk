################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../mymalloc.c \
../tree.c 

OBJS += \
./mymalloc.o \
./tree.o 

C_DEPS += \
./mymalloc.d \
./tree.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -m64 -gencode arch=compute_30,code=sm_30 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -G -g -O0 -m64 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


