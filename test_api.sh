#!/bin/bash

INPUT_FILE="./attack_tree.json"
API_URL="http://localhost:80"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File $INPUT_FILE not found"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed"
    exit 1
fi

echo "Checking API health..."
health_response=$(curl -s "${API_URL}/health")
if [ "$(echo $health_response | jq -r '.status')" != "healthy" ]; then
    echo "Error: API is not healthy"
    exit 1
fi

echo "Generating test cases..."
test_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @"$INPUT_FILE" \
    "${API_URL}/api/v1/generate-tests")

timestamp=$(date +%Y%m%d_%H%M%S)
output_file="test_cases_${timestamp}.json"
echo "$test_response" | jq '.' > "$output_file"

echo "Test cases generated and saved to $output_file"

echo "Querying vectorstore for relevant documents..."
# vulnerabilities=$(echo "$test_response" | jq -r '.test_cases[].vulnerability_addressed' | sort -u)
# echo $vulnerabilities
# for vuln in $vulnerabilities; do
#     echo "Searching for: $vuln"
#     curl -s -X POST \
#         -H "Content-Type: application/json" \
#         -d "{\"query\": \"$vuln\", \"k\": 5}" \
#         "${API_URL}/api/v1/query-vectorstore" | \
#         jq '.' > "docs_${vuln// /_}_${timestamp}.json"
# done

readarray -t vuln_array < <(echo "$test_response" | jq -r '.test_cases[].vulnerability_addressed' | sort -u)

# Then loop through the array
for vuln in "${vuln_array[@]}"; do
    # Create a safe filename by replacing problematic characters
    safe_filename=$(echo "$vuln" | tr ' /()' '_' | tr -d "'\"")
    
    echo "Searching for: $vuln"
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$vuln\", \"k\": 5}" \
        "${API_URL}/api/v1/query-vectorstore" | \
        jq '.' > "docs_${safe_filename}_${timestamp}.json"
done


echo "API testing complete"