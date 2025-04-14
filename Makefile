FUNC := clang
copt := -c 
OBJ_DIR := ./bin/
FLAGS := -O3 -lm -g -Werror -mavx2 -mfma -march=native -fopenmp

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(CPP_FILES:.cpp=.obj)))

all:
	$(FUNC) ./main.cpp -o ./main.exe $(FLAGS)

clean:
	rm -f ./*.exe