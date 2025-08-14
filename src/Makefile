# RML System Build Makefile
# Builds all core RML system components

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
INCLUDES = -I.
LIBS = -ljsoncpp -lpthread

# Directories
SRC_DIR = .
BUILD_DIR = ../build
BIN_DIR = ../bin

# Source files
REASONER_SOURCES = reasoner/graph_engine.cpp
RESPONDER_SOURCES = responder/response_generator.cpp
INTERFACE_SOURCES = interface/cli_interface.cpp
OPTIMIZER_SOURCES = optimizer/*.cpp

# Executables
REASONER_EXE = $(BIN_DIR)/rml_reasoner
RESPONDER_EXE = $(BIN_DIR)/rml_responder
CLI_EXE = $(BIN_DIR)/rml_cli
SYSTEM_EXE = $(BIN_DIR)/rml_system

# Default target
all: directories $(REASONER_EXE) $(RESPONDER_EXE) $(CLI_EXE) $(SYSTEM_EXE)

# Create necessary directories
directories:
	@echo "📁 Creating build directories..."
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build reasoner component
$(REASONER_EXE): $(REASONER_SOURCES)
	@echo "🧠 Building RML Reasoner..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)
	@echo "✅ RML Reasoner built successfully"

# Build responder component
$(RESPONDER_EXE): $(RESPONDER_SOURCES)
	@echo "💬 Building RML Responder..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)
	@echo "✅ RML Responder built successfully"

# Build CLI interface
$(CLI_EXE): $(INTERFACE_SOURCES)
	@echo "🖥️ Building RML CLI Interface..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)
	@echo "✅ RML CLI Interface built successfully"

# Build complete RML system
$(SYSTEM_EXE): $(REASONER_SOURCES) $(RESPONDER_SOURCES) $(INTERFACE_SOURCES)
	@echo "🚀 Building Complete RML System..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)
	@echo "✅ Complete RML System built successfully"

# Test targets
test-reasoner: $(REASONER_EXE)
	@echo "🧪 Testing RML Reasoner..."
	./$(REASONER_EXE)

test-responder: $(RESPONDER_EXE)
	@echo "🧪 Testing RML Responder..."
	./$(RESPONDER_EXE)

test-cli: $(CLI_EXE)
	@echo "🧪 Testing RML CLI Interface..."
	./$(CLI_EXE)

test-system: $(SYSTEM_EXE)
	@echo "🧪 Testing Complete RML System..."
	./$(SYSTEM_EXE)

# Run targets
run-reasoner: $(REASONER_EXE)
	@echo "🚀 Running RML Reasoner..."
	./$(REASONER_EXE)

run-responder: $(RESPONDER_EXE)
	@echo "🚀 Running RML Responder..."
	./$(RESPONDER_EXE)

run-cli: $(CLI_EXE)
	@echo "🚀 Running RML CLI Interface..."
	./$(CLI_EXE)

run-system: $(SYSTEM_EXE)
	@echo "🚀 Running Complete RML System..."
	./$(SYSTEM_EXE)

# Development targets
dev: all
	@echo "🔧 Development build complete"
	@echo "📁 Binaries created in: $(BIN_DIR)"
	@echo "🧪 Run 'make test-system' to test the complete system"

# Clean targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	@echo "✅ Clean complete"

clean-binaries:
	@echo "🧹 Cleaning binaries..."
	rm -f $(BIN_DIR)/*
	@echo "✅ Binaries cleaned"

# Install dependencies (macOS)
install-deps-mac:
	@echo "📦 Installing dependencies on macOS..."
	brew install jsoncpp
	@echo "✅ Dependencies installed"

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	@echo "📦 Installing dependencies on Ubuntu/Debian..."
	sudo apt-get update
	sudo apt-get install -y libjsoncpp-dev
	@echo "✅ Dependencies installed"

# Show help
help:
	@echo "🎯 RML System Build Makefile"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all RML system components"
	@echo "  dev              - Development build with status"
	@echo ""
	@echo "Component builds:"
	@echo "  $(REASONER_EXE)  - Build RML Reasoner"
	@echo "  $(RESPONDER_EXE) - Build RML Responder"
	@echo "  $(CLI_EXE)       - Build RML CLI Interface"
	@echo "  $(SYSTEM_EXE)    - Build Complete RML System"
	@echo ""
	@echo "Testing:"
	@echo "  test-reasoner    - Test RML Reasoner"
	@echo "  test-responder   - Test RML Responder"
	@echo "  test-cli         - Test RML CLI Interface"
	@echo "  test-system      - Test Complete RML System"
	@echo ""
	@echo "Running:"
	@echo "  run-reasoner     - Run RML Reasoner"
	@echo "  run-responder    - Run RML Responder"
	@echo "  run-cli          - Run RML CLI Interface"
	@echo "  run-system       - Run Complete RML System"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean            - Clean all build artifacts"
	@echo "  clean-binaries   - Clean only binaries"
	@echo "  install-deps-mac - Install dependencies (macOS)"
	@echo "  install-deps-ubuntu - Install dependencies (Ubuntu)"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  make install-deps-mac    # Install dependencies"
	@echo "  make all                 # Build everything"
	@echo "  make test-system         # Test the system"
	@echo "  make run-cli             # Run the CLI interface"

# Phony targets
.PHONY: all directories test-reasoner test-responder test-cli test-system \
        run-reasoner run-responder run-cli run-system dev clean clean-binaries \
        install-deps-mac install-deps-ubuntu help 