# GPT-2 Text Generator

## Overview
This project is a web-based text generation application powered by a fine-tuned GPT-2 model. It allows users to input prompts and receive AI-generated text responses via a sleek React-based front-end interface. The back-end is built using Flask, serving a fine-tuned GPT-2 model that processes questions and generates answers using nucleus sampling.
<img width="2702" height="1464" alt="image" src="https://github.com/user-attachments/assets/a194e18f-9ed6-4b44-9dbe-2215b8ef03ed" />


## Features
- **Fine-Tuned GPT-2 Model**: Utilizes a pre-trained GPT-2 model, fine-tuned for question-answering tasks.
- **React Front-End**: A modern, responsive UI built with React and styled with Tailwind CSS, allowing users to input prompts and view generated text.
- **Flask Back-End**: A lightweight Flask server handles API requests, processes input prompts, and returns AI-generated responses.
- **Nucleus Sampling**: Implements nucleus sampling (p=0.7) for generating diverse and coherent text outputs.
- **Cross-Platform Compatibility**: Supports running on CPU or MPS (Apple Silicon) devices for model inference.

## Prerequisites
To run this project, ensure you have the following installed:
- Python 3.8+
- Node.js 16+
- PyTorch
- Flask
- Tiktoken
- Axios (for React front-end)
- A fine-tuned GPT-2 model weights file (`gpt2_finetuned_qa_final.pth`)

## Installation

### Back-End Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedalii3/gpt2-text-generator.git
   cd gpt2-text-generator
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required Python dependencies:
   ```bash
   pip install torch flask flask-cors tiktoken transformers numpy
   ```
4. Place the fine-tuned model weights (`gpt2_finetuned_qa_final.pth`) in the project root or update the `weights_path` in the Python script to the correct location.
5. Run the Flask server:
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5001`.

### Front-End Setup
1. Navigate to the front-end directory (if separate, or create a `frontend` folder):
   ```bash
   cd frontend
   ```
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the React development server:
   ```bash
   npm start
   ```
   The front-end will be available at `http://localhost:3000`.

## Usage
1. Open the React application in your browser (`http://localhost:3000`).
2. Enter a prompt in the text area (e.g., "What is the capital of France?").
3. Click the "Generate Text" button to receive an AI-generated response.
4. View the generated text in the output section, with an option to copy it to the clipboard.
5. The back-end processes the prompt using the fine-tuned GPT-2 model and returns the response via the `/answer` API endpoint.

## Project Structure
```plaintext
gpt2-text-generator/
├── app.py                  # Flask back-end with GPT-2 model
├── frontend/               # React front-end
│   ├── src/
│   │   ├── App.js         # Main React component
│   │   ├── App.css        # Styles for the React app
│   ├── package.json       # Node.js dependencies
├── gpt2_finetuned_qa_final.pth  # Fine-tuned model weights
├── README.md              # This file
```

## API Endpoints
- **POST /answer**: Accepts a JSON payload with a `question` field and returns a JSON response with the generated `answer`.
  - Example Request:
    ```json
    {
      "question": "What is the capital of France?"
    }
    ```
  - Example Response:
    ```json
    {
      "answer": "The capital of France is Paris."
    }
    ```

## Notes
- The fine-tuned model weights (`gpt2_finetuned_qa_final.pth`) must be available and correctly referenced in the `app.py` script.
- The application supports MPS (Apple Silicon) for faster inference if available; otherwise, it falls back to CPU.
- The front-end uses Tailwind CSS via CDN for styling and Axios for API requests.
- The model uses nucleus sampling with `p=0.7` to ensure diverse yet coherent text generation.

## Limitations
<img width="2702" height="1466" alt="image" src="https://github.com/user-attachments/assets/61540430-3ac2-48a8-a314-f23865fa9628" />

- The application does not support queries about personal information (e.g., "What is my name?") and will return a predefined response for such cases.
- The fine-tuned model is optimized for question-answering tasks; performance may vary for other types of prompts.
- The maximum prompt length in the front-end is limited to 200 characters.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on the [GitHub repository](https://github.com/ahmedalii3/gpt2-text-generator).

## Author
- **Ahmed Ali**
  - Email: [ahmed.alii@gmail.com](mailto:ahmed.alii@gmail.com)
  - LinkedIn: [Ahmed Ali](https://www.linkedin.com/in/ahmed-ali-b4baa9203/)
  - GitHub: [ahmedalii3](https://github.com/ahmedalii3)


