TOPDIR = ./
CROSS_COMPILE:=

CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
STRIP = $(CROSS_COMPILE)strip

CXXFLAGS := -Wall -g -std=c++11 -lrt -pthread -mavx -ldl

INCLUDES := -I$(TOPDIR)/include -I../../../ThirdParty/opencv_pc_5.4/include

LIBS := -L$(TOPDIR)/lib -L../../../ThirdParty/opencv_pc_5.4/lib -Wl,-whole-archive -lmagikExecutor -Wl,-no-whole-archive -lprotobuf -lprotobuf-lite -Wl,-rpath ./lib -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs -Wl,-rpath ../../../ThirdParty/opencv_pc_5.4/lib

TARGET = pc_inference_bin

OBJS :=inference.o

%.o:%.cc
	$(CPP) $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

$(TARGET):$(OBJS)
	$(CPP) $(CXXFLAGS) $(OBJS) -o $@ $(INCLUDES) $(LIBS)

all:$(TARGET)

.PHONY: clean
clean: 
	rm -f $(TARGET) $(OBJS)
