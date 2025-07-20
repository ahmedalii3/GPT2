import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [generatedTexts, setGeneratedTexts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5001/answer', {
        question: prompt
    });
    setGeneratedTexts([response.data.answer]);
    } catch (err) {
      console.error('API error:', err);
      setError('Failed to generate text: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => alert('Copied to clipboard!'));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-blue-900 text-white flex flex-col items-center justify-center pt-12 p-4">
      <header className="w-full max-w-2xl mb-6">
        <h1 className="text-4xl font-bold text-center text-blue-200 mb-2">GPT-2 Text Generator</h1>
        <p className="text-center text-gray-400 text-sm">Create unique text with AI</p>
      </header>
      <div className="w-full max-w-2xl bg-gray-800/90 backdrop-blur-md rounded-xl shadow-2xl p-6 transform transition-all duration-300 hover:shadow-3xl">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="prompt" className="block text-sm font-medium text-gray-300 mb-2">
              Enter your prompt:
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full h-40 p-4 border border-gray-600 rounded-lg bg-gray-700 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              placeholder="Type your prompt here (e.g., 'Once upon a time...')"
              maxLength={200}
              required
            />
            <p className="text-right text-xs text-gray-500 mt-1">{prompt.length}/200</p>
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {loading ? (
              <span className="flex items-center">
                <svg className="animate-spin h-5 w-5 mr-2 text-white" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating...
              </span>
            ) : (
              <>
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 3a7 7 0 100 14 7 7 0 000-14zm-1 10V7h2v6h-2zm1-8a1 1 0 110-2 1 1 0 010 2z"/>
                </svg>
                Generate Text
              </>
            )}
          </button>
        </form>
        {error && (
          <p className="mt-4 text-red-400 text-center font-medium">{error}</p>
        )}
        {generatedTexts.length > 0 && (
          <div className="mt-6">
            <h2 className="text-xl font-semibold text-gray-200 mb-4">Generated Texts:</h2>
            <div className="space-y-3">
              {generatedTexts.map((text, index) => (
                <div key={index} className="p-4 bg-gray-700 rounded-lg border border-gray-600 flex justify-between items-start">
                  <p className="text-gray-300 flex-1 mr-4">{text}</p>
                  <button
                    onClick={() => copyToClipboard(text)}
                    className="text-blue-400 hover:text-blue-300 transition-colors"
                  >
                    Copy
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      <footer className="mt-10 text-center text-sm text-gray-500">
        <p>Made by Ahmed Ali</p>
        <div className="flex justify-center space-x-4 mt-2">
          <a
            href="mailto:ahmed.alii@gmail.com"
            className="hover:text-white transition-colors"
            target="_blank"
            rel="noopener noreferrer"
          >
            Email
          </a>
          <a
            href="https://www.linkedin.com/in/ahmed-ali-b4baa9203/"
            className="hover:text-white transition-colors"
            target="_blank"
            rel="noopener noreferrer"
          >
            LinkedIn
          </a>
          <a
            href="https://github.com/ahmedalii3"
            className="hover:text-white transition-colors"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </div>
      </footer>
    </div>
  
  );
}

export default App;