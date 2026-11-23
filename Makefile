CXX ?= c++
CXXFLAGS ?= -O2 -g
# override keeps required flags when CXXFLAGS/CPPFLAGS are set on the command line.
override CXXFLAGS += -std=c++20 -Wall -Wextra
# Emit .d files so edits to headers rebuild the objects that include them.
override CPPFLAGS += -MMD -MP

BUILD_DIR = build
TARGET = $(BUILD_DIR)/main
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)
LIB_OBJECTS = $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS))
TEST_SOURCES = $(wildcard tests/*_test.cpp)
TEST_TARGETS = $(TEST_SOURCES:tests/%.cpp=$(BUILD_DIR)/%)
TEST_OBJECTS = $(TEST_SOURCES:tests/%.cpp=$(BUILD_DIR)/tests/%.o)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%_test: $(BUILD_DIR)/tests/%_test.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Test objects are built only through the pattern chain above, so make would delete them as intermediates.
.SECONDARY: $(TEST_OBJECTS)

$(BUILD_DIR)/tests/%.o: tests/%.cpp | $(BUILD_DIR)/tests
	$(CXX) $(CPPFLAGS) -Isrc $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR) $(BUILD_DIR)/tests:
	mkdir -p $@

.PHONY: run test clean

run: $(TARGET)
	./$(TARGET)

test: $(TEST_TARGETS)
	@for t in $(TEST_TARGETS); do echo "== $$t"; ./$$t ./models/stories15M.bin || exit 1; done

clean:
	$(RM) -r $(BUILD_DIR)

-include $(OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d)
