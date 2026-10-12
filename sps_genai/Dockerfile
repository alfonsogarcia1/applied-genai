# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock /code/

# Install only what the API needs (skip the notebook and dev groups)
RUN uv sync --frozen --no-install-project --no-default-groups

# Copy the application code (app/ includes the trained classifier weights)
COPY ./app /code/app
COPY ./helper_lib /code/helper_lib

# Command to run the application
CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--port", "80"]
