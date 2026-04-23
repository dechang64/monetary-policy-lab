#!/bin/bash
# ─────────────────────────────────────────────────────────────
# deploy.sh — One-click deployment for Monetary Policy Research Lab
#
# Usage:
#   ./deploy.sh              # Deploy with Docker
#   ./deploy.sh --local      # Run locally (no Docker)
#   ./deploy.sh --cloud      # Deploy to Streamlit Community Cloud
# ─────────────────────────────────────────────────────────────

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Monetary Policy Research Lab — Deploy Script   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

MODE="${1:---docker}"

# ── Check prerequisites ──
check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}✗ $1 not found. Please install: $2${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ $1 found${NC}"
}

case "$MODE" in
    --local)
        echo -e "${BLUE}Mode: Local deployment (no Docker)${NC}"
        check_cmd python3 "python3"
        check_cmd pip3 "pip3"

        echo ""
        echo -e "${BLUE}Installing dependencies...${NC}"
        pip3 install -r requirements.txt -q

        echo ""
        echo -e "${GREEN}Starting Monetary Policy Research Lab...${NC}"
        echo -e "${GREEN}Open http://localhost:8501 in your browser${NC}"
        echo ""
        streamlit run app.py \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --browser.gatherUsageStats false
        ;;

    --docker)
        echo -e "${BLUE}Mode: Docker deployment${NC}"
        check_cmd docker "Docker (https://docs.docker.com/get-docker/)"

        # Check for docker compose (plugin or standalone)
        if docker compose version &> /dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
        elif command -v docker-compose &> /dev/null 2>&1; then
            COMPOSE_CMD="docker-compose"
        else
            echo -e "${RED}✗ Docker Compose not found. Install Docker Compose plugin: https://docs.docker.com/compose/install/${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Docker Compose found (${COMPOSE_CMD})${NC}"

        # Check .env
        if [ ! -f .env ]; then
            cp .env.example .env
            echo -e "${BLUE}Created .env from template${NC}"
            echo -e "${RED}⚠  Edit .env and add your FRED_API_KEY (optional)${NC}"
            echo ""
        fi

        echo ""
        echo -e "${BLUE}Building Docker image...${NC}"
        $COMPOSE_CMD build --quiet

        echo ""
        echo -e "${GREEN}Starting container...${NC}"
        $COMPOSE_CMD up -d

        echo ""
        echo -e "${GREEN}✓ Deployed! Open http://localhost:8501${NC}"
        echo -e "${BLUE}To stop: ${COMPOSE_CMD} down${NC}"
        echo -e "${BLUE}To view logs: ${COMPOSE_CMD} logs -f${NC}"
        ;;

    --cloud)
        echo -e "${BLUE}Mode: Streamlit Community Cloud deployment${NC}"
        echo ""
        echo "Steps to deploy:"
        echo ""
        echo "  1. Create a GitHub repository"
        echo "     git init && git add . && git commit -m 'init'"
        echo "     git remote add origin https://github.com/YOUR_USERNAME/mp-research-lab.git"
        echo "     git push -u origin main"
        echo ""
        echo "  2. Go to https://share.streamlit.io/"
        echo "     - Sign in with GitHub"
        echo "     - Click 'New app'"
        echo "     - Select your repository"
        echo "     - Main file path: app.py"
        echo ""
        echo "  3. (Optional) Add FRED_API_KEY in Secrets"
        echo "     - In the app settings → Secrets"
        echo "     - Add: FRED_API_KEY = your_key_here"
        echo ""
        echo "  4. Your app will be live at: https://your-app.streamlit.app"
        echo ""
        echo -e "${GREEN}Note: Users can also enter their own FRED key in the app${NC}"
        ;;

    *)
        echo "Usage: $0 [--local|--docker|--cloud]"
        exit 1
        ;;
esac
