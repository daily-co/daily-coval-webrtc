# daily-coval-webrtc

[![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.daily.co) [![Discord](https://img.shields.io/discord/1217145424381743145)](https://discord.gg/dailyco)

A template voice agent for [Pipecat Cloud](https://www.daily.co/products/pipecat-cloud/) that demonstrates building and deploying a conversational AI agent.

## Prerequisites

- Python 3.10+
- Linux, MacOS, or Windows Subsystem for Linux (WSL)
- [Docker](https://www.docker.com) and a Docker repository (e.g., [Docker Hub](https://hub.docker.com))
- A Docker Hub account (or other container registry account)
- [Pipecat Cloud](https://pipecat.daily.co) account

> **Note**: If you haven't installed Docker yet, follow the official installation guides for your platform ([Linux](https://docs.docker.com/engine/install/), [Mac](https://docs.docker.com/desktop/setup/install/mac-install/), [Windows](https://docs.docker.com/desktop/setup/install/windows-install/)). For Docker Hub, [create a free account](https://hub.docker.com/signup) and log in via terminal with `docker login`.

## Get Started

### 1. Get the starter project

Clone the starter project from GitHub:

```bash
git clone https://github.com/daily-co/pipecat-cloud-starter
cd pipecat-cloud-starter
```

### 2. Set up your Python environment

We recommend using a virtual environment to manage your Python dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the Pipecat Cloud CLI
pip install pipecatcloud
```

### 3. Authenticate with Pipecat Cloud

```bash
pcc auth login
```

### 4. Acquire required API keys

This starter requires the following API keys:

- **OpenAI API Key**: Get from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Cartesia API Key**: Get from [play.cartesia.ai/keys](https://play.cartesia.ai/keys)
- **Daily API Key**: Automatically provided through your Pipecat Cloud account

### 5. Configure to run locally (optional)

You can test your agent locally before deploying to Pipecat Cloud:

```bash
# Set environment variables with your API keys
export CARTESIA_API_KEY="your_cartesia_key"
export DAILY_API_KEY="your_daily_key"
export OPENAI_API_KEY="your_openai_key"
```

> Your `DAILY_API_KEY` can be found at [https://pipecat.daily.co](https://pipecat.daily.co) under the `Settings` in the `Daily (WebRTC)` tab.

First install requirements:

```bash
pip install -r requirements.txt
```

Then, launch the bot.py script locally:

```bash
python bot.py
```

## Deploy & Run

### 1. Build and push your Docker image

```bash
# Build the image (targeting ARM architecture for cloud deployment)
docker build --platform=linux/arm64 -t my-first-agent:latest .

# Tag with your Docker username and version
docker tag my-first-agent:latest your-username/my-first-agent:0.1

# Push to Docker Hub
docker push your-username/my-first-agent:0.1
```

### 2. Create a secret set for your API keys

The starter project requires API keys for OpenAI and Cartesia:

```bash
# Copy the example env file
cp env.example .env

# Edit .env to add your API keys:
# CARTESIA_API_KEY=your_cartesia_key
# OPENAI_API_KEY=your_openai_key

# Create a secret set from your .env file
pcc secrets set my-first-agent-secrets --file .env
```

Alternatively, you can create secrets directly via CLI:

```bash
pcc secrets set my-first-agent-secrets \
  CARTESIA_API_KEY=your_cartesia_key \
  OPENAI_API_KEY=your_openai_key
```

### 3. Deploy to Pipecat Cloud

```bash
pcc deploy my-first-agent your-username/my-first-agent:0.1 --secrets my-first-agent-secrets
```

> **Note (Optional)**: For a more maintainable approach, you can use the included `pcc-deploy.toml` file:
>
> ```toml
> agent_name = "my-first-agent"
> image = "your-username/my-first-agent:0.1"
> secret_set = "my-first-agent-secrets"
>
> [scaling]
>     min_instances = 0
> ```
>
> Then simply run `pcc deploy` without additional arguments.

> **Note**: If your repository is private, you'll need to add credentials:
>
> ```bash
> # Create pull secret (you’ll be prompted for credentials)
> pcc secrets image-pull-secret pull-secret https://index.docker.io/v1/
>
> # Deploy with credentials
> pcc deploy my-first-agent your-username/my-first-agent:0.1 --credentials pull-secret
> ```

### 4. Check deployment and scaling (optional)

By default, your agent will use "scale-to-zero" configuration, which means it may have a cold start of around 10 seconds when first used. By default, idle instances are maintained for 5 minutes before being terminated when using scale-to-zero.

For more responsive testing, you can scale your deployment to keep a minimum of one instance warm:

```bash
# Ensure at least one warm instance is always available
pcc deploy my-first-agent your-username/my-first-agent:0.1 --min-instances 1

# Check the status of your deployment
pcc agent status my-first-agent
```

By default, idle instances are maintained for 5 minutes before being terminated when using scale-to-zero.

### 5. Create an API key

```bash
# Create a public API key for accessing your agent
pcc organizations keys create

# Set it as the default key to use with your agent
pcc organizations keys use
```

### 6. Start your agent

```bash
# Start a session with your agent in a Daily room
pcc agent start my-first-agent --use-daily
```

This will return a URL, which you can use to connect to your running agent.

## Documentation

For more details on Pipecat Cloud and its capabilities:

- [Pipecat Cloud Documentation](https://docs.pipecat.daily.co)
- [Pipecat Project Documentation](https://docs.pipecat.ai)

## Running a deployed agent for testing
generate a url to interact with the agent
```
curl -s --request POST \
--url "https://api.pipecat.daily.co/v1/public/daily-coval-webrtc/start" \
--header "authorization: Bearer $PCC_API_KEY" \
--header 'content-type: application/json' \
--data '{
  "createDailyRoom": true,
  "dailyRoomProperties": { "start_video_off": true, "enable_chat": true },
  "body": {"coval": true}
}' |jq -r '(.dailyRoom | tostring) + "?t=" + (.dailyToken | tostring)'
```
==>
https://cloud-domain4565.daily.co/randomroomname?t=tokeneyJhbGciOi...

## ignore

### notes to self
```
docker build --platform=linux/arm64 -t daily-coval-webrtc:latest .
docker tag daily-coval-webrtc:latest vipipecat/daily-coval-webrtc:0.2
docker push vipipecat/daily-coval-webrtc:0.2

pcc secrets set daily-coval-webrtc-agent-secrets --file .env

pcc deploy daily-coval-webrtc \
vipipecat/daily-coval-webrtc:0.2 \
--min-instances 1 \
--secrets daily-coval-webrtc-agent-secrets

pcc agent status daily-coval-webrtc

pcc organizations keys create
pcc organizations keys use

pcc agent start daily-coval-webrtc
```

