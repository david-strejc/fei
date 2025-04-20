#!/bin/bash
# Script to manage API keys for the fei assistant

# Function to show the current API keys
show_keys() {
    echo "Current API keys:"
    echo "-----------------"
    
    # Check environment variables
    echo "Environment variables:"
    echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-<not set>}"
    echo "OPENAI_API_KEY=${OPENAI_API_KEY:-<not set>}"
    echo "GROQ_API_KEY=${GROQ_API_KEY:-<not set>}"
    echo "GOOGLE_API_KEY=${GOOGLE_API_KEY:-<not set>}"
    echo "BRAVE_API_KEY=${BRAVE_API_KEY:-<not set>}"
    echo
    
    # Check .env file
    if [ -f .env ]; then
        echo ".env file contents:"
        grep -E "^[A-Z_]+=.+" .env | sed 's/=.*$/=********/'
        echo
    else
        echo "No .env file found."
        echo
    fi
    
    # Check ~/.fei.ini file
    if [ -f ~/.fei.ini ]; then
        echo "~/.fei.ini file contents:"
        grep -A 1 "\[anthropic\]" ~/.fei.ini | grep "api_key" | sed 's/=.*$/= ********/'
        grep -A 1 "\[openai\]" ~/.fei.ini | grep "api_key" | sed 's/=.*$/= ********/'
        grep -A 1 "\[groq\]" ~/.fei.ini | grep "api_key" | sed 's/=.*$/= ********/'
        grep -A 1 "\[google\]" ~/.fei.ini | grep "api_key" | sed 's/=.*$/= ********/'
        grep -A 1 "\[brave\]" ~/.fei.ini | grep "api_key" | sed 's/=.*$/= ********/'
        echo
    else
        echo "No ~/.fei.ini file found."
        echo
    fi
}

# Function to update the API key in the .env file
update_env_key() {
    local provider=$1
    local key=$2
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        echo "# API Keys" > .env
    fi
    
    # Check if the key already exists in the .env file
    if grep -q "^${provider}_API_KEY=" .env; then
        # Update the existing key
        sed -i "s|^${provider}_API_KEY=.*|${provider}_API_KEY=${key}|" .env
    else
        # Add the new key
        echo "${provider}_API_KEY=${key}" >> .env
    fi
    
    echo "${provider}_API_KEY updated in .env file."
}

# Function to update the API key in the ~/.fei.ini file
update_ini_key() {
    local provider=$1
    local key=$2
    
    # Create ~/.fei.ini file if it doesn't exist
    if [ ! -f ~/.fei.ini ]; then
        echo "[api]" > ~/.fei.ini
        echo "" >> ~/.fei.ini
        echo "[anthropic]" >> ~/.fei.ini
        echo "" >> ~/.fei.ini
        echo "[openai]" >> ~/.fei.ini
        echo "" >> ~/.fei.ini
        echo "[groq]" >> ~/.fei.ini
        echo "" >> ~/.fei.ini
        echo "[google]" >> ~/.fei.ini
        echo "" >> ~/.fei.ini
        echo "[brave]" >> ~/.fei.ini
        echo "" >> ~/.fei.ini
        echo "[mcp]" >> ~/.fei.ini
        echo "" >> ~/.fei.ini
        echo "[user]" >> ~/.fei.ini
        echo "" >> ~/.fei.ini
    fi
    
    # Check if the section exists in the ~/.fei.ini file
    if ! grep -q "^\[${provider,,}\]" ~/.fei.ini; then
        # Add the section
        echo "" >> ~/.fei.ini
        echo "[${provider,,}]" >> ~/.fei.ini
        echo "" >> ~/.fei.ini
    fi
    
    # Check if the key already exists in the section
    if grep -A 5 "^\[${provider,,}\]" ~/.fei.ini | grep -q "^api_key"; then
        # Update the existing key
        sed -i "/^\[${provider,,}\]/,/^\[/ s|^api_key.*|api_key = ${key}|" ~/.fei.ini
    else
        # Add the new key
        sed -i "/^\[${provider,,}\]/a api_key = ${key}" ~/.fei.ini
    fi
    
    echo "${provider} API key updated in ~/.fei.ini file."
}

# Function to run fei with the specified API key
run_fei() {
    local provider=$1
    local key=$2
    shift 2
    
    # Unset any existing API key environment variables
    unset ANTHROPIC_API_KEY
    unset OPENAI_API_KEY
    unset GROQ_API_KEY
    unset GOOGLE_API_KEY
    unset BRAVE_API_KEY
    
    # Set the specified API key
    export ${provider}_API_KEY="${key}"
    
    # Run fei with the remaining arguments
    python -m fei "$@"
}

# Main script
case "$1" in
    show)
        show_keys
        ;;
    update)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 update <provider> <key>"
            echo "Providers: ANTHROPIC, OPENAI, GROQ, GOOGLE, BRAVE"
            exit 1
        fi
        
        update_env_key "$2" "$3"
        update_ini_key "$2" "$3"
        ;;
    run)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 run <provider> <key> [fei arguments]"
            echo "Providers: ANTHROPIC, OPENAI, GROQ, GOOGLE, BRAVE"
            exit 1
        fi
        
        run_fei "$2" "$3" "${@:4}"
        ;;
    *)
        echo "Usage: $0 {show|update|run}"
        echo "  show                        Show current API keys"
        echo "  update <provider> <key>     Update API key for provider"
        echo "  run <provider> <key> [args] Run fei with specified API key"
        echo
        echo "Providers: ANTHROPIC, OPENAI, GROQ, GOOGLE, BRAVE"
        exit 1
        ;;
esac
