
## Development Workflow

For local development without Docker, you can use `uv` for environment and dependency management.

1.  **Install uv** (if not already installed)
    ```bash
    pip install uv
    ```

2.  **Create a virtual environment with Python 3.12**
    ```bash
    uv venv --python=python3.11
    ```

3.  **Activate the virtual environment**

    Windows:
    ```bash
    .venv\Scripts\activate
    ```

    Linux/macOS:
    ```bash
    source .venv/bin/activate
    ```

4.  **Install dependencies with uv**
    ```bash
    uv sync
    ```

5.  **Install the pre-commit hooks**
    ```bash
    pre-commit install
    ```

6.  **Running the Application Locally**
    To run the development server locally:
    ```bash
    uvicorn app.main:app --reload
    ```

7.  **Using Ruff for linting and formatting**
    ```bash
    # Run linting
    ruff check .

    # Apply automatic fixes to linting issues
    ruff check --fix .

    # Format code
    ruff format .
    ```

## License

Free
