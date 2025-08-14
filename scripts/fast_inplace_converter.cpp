#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <regex>
#include <filesystem>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

class FastInplaceConverter {
private:
    std::string data_dir;
    std::atomic<int> files_processed{0};
    std::atomic<int> files_converted{0};
    std::atomic<int> total_samples{0};
    std::atomic<int> conversion_errors{0};
    std::mutex output_mutex;
    
    // RML components
    std::vector<std::string> rml_components = {
        "concepts", "triples", "entities", "emotions", 
        "reasoning", "intents", "summaries", "events", "vectors", "tags"
    };
    
    // Storage monitoring
    std::atomic<long long> total_storage_used{0};
    
public:
    FastInplaceConverter(const std::string& dir) : data_dir(dir) {}
    
    struct FileAnalysis {
        std::string type;
        std::string file_type;
        std::string message;
        std::vector<std::string> keys;
        std::map<std::string, std::string> sample;
        int line_count;
        long long file_size;
    };
    
    void run() {
        std::cout << "🚀 Starting FAST in-place data conversion..." << std::endl;
        std::cout << "📁 Data directory: " << data_dir << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Find all JSONL files
        std::vector<fs::path> jsonl_files;
        find_jsonl_files(data_dir, jsonl_files);
        
        std::cout << "📁 Found " << jsonl_files.size() << " JSONL files to process" << std::endl;
        
        // Process files
        for (size_t i = 0; i < jsonl_files.size(); ++i) {
            const auto& file_path = jsonl_files[i];
            process_file(file_path, i + 1, jsonl_files.size());
            
            // Monitor storage every 10 files
            if ((i + 1) % 10 == 0) {
                monitor_storage();
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        print_final_statistics(duration.count());
    }
    
private:
    void find_jsonl_files(const std::string& dir, std::vector<fs::path>& files) {
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
                files.push_back(entry.path());
            }
        }
    }
    
    void process_file(const fs::path& file_path, int file_num, int total_files) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "📋 Processing file " << file_num << "/" << total_files << ": " << file_path.filename() << std::endl;
        std::cout << "📁 Full path: " << file_path << std::endl;
        
        // Analyze file first
        auto analysis = analyze_file(file_path);
        
        if (analysis.type == "error") {
            std::cout << "❌ Cannot read file: " << analysis.message << std::endl;
            conversion_errors++;
            return;
        }
        
        if (analysis.type == "empty") {
            std::cout << "⚠️ File is empty, skipping" << std::endl;
            return;
        }
        
        // Convert based on file type
        int samples_converted = 0;
        
        if (analysis.file_type == "complete_rml") {
            samples_converted = convert_complete_rml_file(file_path, analysis);
        } else if (analysis.file_type == "component_file") {
            samples_converted = convert_component_file(file_path, analysis);
        } else if (analysis.file_type == "mixed_data") {
            samples_converted = convert_mixed_data_file(file_path, analysis);
        }
        
        // Update statistics
        files_processed++;
        if (samples_converted > 0) {
            files_converted++;
            total_samples += samples_converted;
        }
        
        std::cout << "📊 File summary: " << samples_converted << " samples converted" << std::endl;
    }
    
    FileAnalysis analyze_file(const fs::path& file_path) {
        std::cout << "🔍 Analyzing: " << file_path.filename() << std::endl;
        
        FileAnalysis analysis;
        analysis.file_size = fs::file_size(file_path);
        
        std::ifstream file(file_path);
        if (!file.is_open()) {
            analysis.type = "error";
            analysis.message = "Cannot open file";
            return analysis;
        }
        
        std::string line;
        std::vector<std::string> lines;
        int line_count = 0;
        
        // Read first 10 lines for analysis
        while (std::getline(file, line) && line_count < 10) {
            if (!line.empty()) {
                lines.push_back(line);
            }
            line_count++;
        }
        
        if (lines.empty()) {
            analysis.type = "empty";
            analysis.message = "File is empty";
            return analysis;
        }
        
        // Try to parse first line as JSON
        try {
            auto sample_data = parse_json_line(lines[0]);
            analysis.keys = get_keys(sample_data);
            analysis.sample = sample_data;
            analysis.line_count = lines.size();
            
            // Determine file type
            if (has_all_rml_components(analysis.keys)) {
                analysis.file_type = "complete_rml";
            } else if (analysis.keys.size() <= 3) {
                analysis.file_type = "component_file";
            } else {
                analysis.file_type = "mixed_data";
            }
            
            analysis.type = "json";
            
        } catch (const std::exception& e) {
            analysis.type = "invalid_json";
            analysis.message = "Not valid JSON";
        }
        
        return analysis;
    }
    
    std::map<std::string, std::string> parse_json_line(const std::string& line) {
        std::map<std::string, std::string> result;
        
        // Simple JSON parser for key-value pairs
        std::regex key_value_pattern("\"([^\"]+)\"\\s*:\\s*\"([^\"]*)\"");
        std::sregex_iterator iter(line.begin(), line.end(), key_value_pattern);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            std::string key = iter->str(1);
            std::string value = iter->str(2);
            result[key] = value;
        }
        
        return result;
    }
    
    std::vector<std::string> get_keys(const std::map<std::string, std::string>& data) {
        std::vector<std::string> keys;
        for (const auto& pair : data) {
            keys.push_back(pair.first);
        }
        return keys;
    }
    
    bool has_all_rml_components(const std::vector<std::string>& keys) {
        for (const auto& component : rml_components) {
            bool found = false;
            for (const auto& key : keys) {
                if (key == component) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }
    
    int convert_complete_rml_file(const fs::path& file_path, const FileAnalysis& analysis) {
        std::cout << "✅ Converting complete RML file: " << file_path.filename() << std::endl;
        
        int converted_samples = 0;
        std::vector<std::string> converted_lines;
        
        std::ifstream file(file_path);
        std::string line;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            try {
                auto data = parse_json_line(line);
                auto cleaned_data = clean_rml_data(data);
                
                if (!cleaned_data.empty()) {
                    std::string converted_line = convert_to_json(cleaned_data);
                    converted_lines.push_back(converted_line);
                    converted_samples++;
                }
                
            } catch (const std::exception& e) {
                // Skip invalid lines
                continue;
            }
        }
        
        // Write back to file (in-place conversion)
        if (!converted_lines.empty()) {
            std::ofstream out_file(file_path);
            for (const auto& converted_line : converted_lines) {
                out_file << converted_line << std::endl;
            }
        }
        
        std::cout << "  ✅ Converted " << converted_samples << " samples" << std::endl;
        return converted_samples;
    }
    
    int convert_component_file(const fs::path& file_path, const FileAnalysis& analysis) {
        std::cout << "🔧 Converting component file: " << file_path.filename() << std::endl;
        
        std::string component_type = identify_component_type(analysis.sample);
        
        if (component_type.empty()) {
            std::cout << "  ⚠️ Could not identify component type" << std::endl;
            return 0;
        }
        
        std::cout << "  📋 Identified as: " << component_type << std::endl;
        
        int converted_samples = 0;
        std::vector<std::string> converted_lines;
        
        std::ifstream file(file_path);
        std::string line;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            try {
                auto data = parse_json_line(line);
                auto converted_component = convert_component_data(data, component_type);
                
                if (!converted_component.empty()) {
                    std::string converted_line = convert_to_json(converted_component);
                    converted_lines.push_back(converted_line);
                    converted_samples++;
                }
                
            } catch (const std::exception& e) {
                continue;
            }
        }
        
        // Write back to file
        if (!converted_lines.empty()) {
            std::ofstream out_file(file_path);
            for (const auto& converted_line : converted_lines) {
                out_file << converted_line << std::endl;
            }
        }
        
        std::cout << "  ✅ Converted " << converted_samples << " " << component_type << " samples" << std::endl;
        return converted_samples;
    }
    
    int convert_mixed_data_file(const fs::path& file_path, const FileAnalysis& analysis) {
        std::cout << "🔄 Converting mixed data file: " << file_path.filename() << std::endl;
        
        int total_extracted = 0;
        std::vector<std::string> converted_lines;
        
        std::ifstream file(file_path);
        std::string line;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            try {
                auto data = parse_json_line(line);
                auto extracted = extract_rml_from_mixed_data(data);
                
                if (!extracted.empty()) {
                    std::string converted_line = convert_to_json(extracted);
                    converted_lines.push_back(converted_line);
                    total_extracted++;
                }
                
            } catch (const std::exception& e) {
                continue;
            }
        }
        
        // Write back to file
        if (!converted_lines.empty()) {
            std::ofstream out_file(file_path);
            for (const auto& converted_line : converted_lines) {
                out_file << converted_line << std::endl;
            }
        }
        
        std::cout << "  📋 Extracted " << total_extracted << " RML samples" << std::endl;
        return total_extracted;
    }
    
    std::string identify_component_type(const std::map<std::string, std::string>& sample_data) {
        for (const auto& component : rml_components) {
            if (sample_data.find(component) != sample_data.end()) {
                return component;
            }
        }
        
        // Check for component indicators
        for (const auto& pair : sample_data) {
            std::string key_lower = pair.first;
            std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
            
            if (key_lower.find("concept") != std::string::npos) return "concepts";
            if (key_lower.find("entity") != std::string::npos) return "entities";
            if (key_lower.find("emotion") != std::string::npos) return "emotions";
            if (key_lower.find("intent") != std::string::npos) return "intents";
            if (key_lower.find("reasoning") != std::string::npos) return "reasoning";
            if (key_lower.find("summary") != std::string::npos) return "summaries";
            if (key_lower.find("event") != std::string::npos) return "events";
            if (key_lower.find("vector") != std::string::npos) return "vectors";
            if (key_lower.find("tag") != std::string::npos) return "tags";
            if (key_lower.find("triple") != std::string::npos || 
                key_lower.find("relation") != std::string::npos) return "triples";
        }
        
        return "";
    }
    
    std::map<std::string, std::string> convert_component_data(const std::map<std::string, std::string>& data, 
                                                             const std::string& component_type) {
        std::map<std::string, std::string> converted;
        converted["component_type"] = component_type;
        
        // Extract metadata
        for (const auto& pair : data) {
            if (pair.first == "record_id" || pair.first == "doc_id" || 
                pair.first == "document_id" || pair.first == "confidence") {
                converted["metadata_" + pair.first] = pair.second;
            }
        }
        
        // Extract component data
        if (data.find(component_type) != data.end()) {
            converted["data"] = data.at(component_type);
        } else if (data.find("data") != data.end()) {
            converted["data"] = data.at("data");
        } else {
            // Use first non-metadata field
            for (const auto& pair : data) {
                if (pair.first.find("metadata_") == 0) continue;
                converted["data"] = pair.second;
                break;
            }
        }
        
        return converted;
    }
    
    std::map<std::string, std::string> extract_rml_from_mixed_data(const std::map<std::string, std::string>& data) {
        std::map<std::string, std::string> extracted;
        
        for (const auto& pair : data) {
            std::string key_lower = pair.first;
            std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
            
            if (key_lower.find("concept") != std::string::npos || 
                key_lower.find("keyword") != std::string::npos) {
                extracted["concepts"] = pair.second;
            } else if (key_lower.find("entity") != std::string::npos || 
                       key_lower.find("person") != std::string::npos) {
                extracted["entities"] = pair.second;
            } else if (key_lower.find("emotion") != std::string::npos || 
                       key_lower.find("sentiment") != std::string::npos) {
                extracted["emotions"] = pair.second;
            } else if (key_lower.find("intent") != std::string::npos || 
                       key_lower.find("purpose") != std::string::npos) {
                extracted["intents"] = pair.second;
            } else if (key_lower.find("reasoning") != std::string::npos || 
                       key_lower.find("logic") != std::string::npos) {
                extracted["reasoning"] = pair.second;
            } else if (key_lower.find("summary") != std::string::npos || 
                       key_lower.find("abstract") != std::string::npos) {
                extracted["summaries"] = pair.second;
            } else if (key_lower.find("event") != std::string::npos || 
                       key_lower.find("action") != std::string::npos) {
                extracted["events"] = pair.second;
            } else if (key_lower.find("vector") != std::string::npos || 
                       key_lower.find("embedding") != std::string::npos) {
                extracted["vectors"] = pair.second;
            } else if (key_lower.find("tag") != std::string::npos || 
                       key_lower.find("label") != std::string::npos) {
                extracted["tags"] = pair.second;
            } else if (key_lower.find("triple") != std::string::npos || 
                       key_lower.find("relation") != std::string::npos) {
                extracted["triples"] = pair.second;
            }
        }
        
        return extracted;
    }
    
    std::map<std::string, std::string> clean_rml_data(const std::map<std::string, std::string>& data) {
        std::map<std::string, std::string> cleaned;
        
        for (const auto& component : rml_components) {
            auto it = data.find(component);
            if (it != data.end() && !it->second.empty()) {
                cleaned[component] = it->second;
            }
        }
        
        return cleaned.size() >= 3 ? cleaned : std::map<std::string, std::string>();
    }
    
    std::string convert_to_json(const std::map<std::string, std::string>& data) {
        std::stringstream ss;
        ss << "{";
        
        bool first = true;
        for (const auto& pair : data) {
            if (!first) ss << ",";
            ss << "\"" << pair.first << "\":\"" << pair.second << "\"";
            first = false;
        }
        
        ss << "}";
        return ss.str();
    }
    
    void monitor_storage() {
        std::lock_guard<std::mutex> lock(output_mutex);
        
        // Calculate current storage usage
        long long current_usage = 0;
        for (const auto& entry : fs::recursive_directory_iterator(data_dir)) {
            if (entry.is_regular_file()) {
                current_usage += fs::file_size(entry.path());
            }
        }
        
        double usage_gb = current_usage / (1024.0 * 1024.0 * 1024.0);
        
        std::cout << "💾 Storage monitoring: " << std::fixed << std::setprecision(2) 
                  << usage_gb << " GB used" << std::endl;
    }
    
    void print_final_statistics(int duration_seconds) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "🎉 FAST IN-PLACE DATA CONVERSION COMPLETE!" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "📊 Files processed: " << files_processed << std::endl;
        std::cout << "✅ Files converted: " << files_converted << std::endl;
        std::cout << "📈 Total samples: " << total_samples << std::endl;
        std::cout << "❌ Conversion errors: " << conversion_errors << std::endl;
        std::cout << "⏱️ Time taken: " << duration_seconds / 60.0 << " minutes" << std::endl;
        std::cout << "📁 Data directory: " << data_dir << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::string data_directory = "data/";
    
    if (argc > 1) {
        data_directory = argv[1];
    }
    
    FastInplaceConverter converter(data_directory);
    converter.run();
    
    return 0;
} 