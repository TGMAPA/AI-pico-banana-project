import { useState, useEffect } from 'react'

/**
 * ImageGenerator Component
 * ========================
 * Main component for model selection and image generation.
 */
function ImageGenerator() {
  // State management
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState(null)
  const [imageData, setImageData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [generationMode, setGenerationMode] = useState(null)
  const [modelsLoading, setModelsLoading] = useState(true)

  // Backend configuration
  const BACKEND_URL = 'http://localhost:5000'
  const MODELS_ENDPOINT = `${BACKEND_URL}/models`
  const GENERATE_ENDPOINT = `${BACKEND_URL}/generate`

  /**
   * Fetch available models from backend on component mount
   */
  useEffect(() => {
    fetchModels()
  }, [])

  /**
   * Automatically select first model when models are loaded
   */
  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      setSelectedModel(models[0].name)
    }
  }, [models])

  /**
   * Fetch list of available models from backend
   */
  const fetchModels = async () => {
    setModelsLoading(true)
    try {
      console.log('Fetching models from backend...')
      
      const response = await fetch(MODELS_ENDPOINT)
      
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.status === 'success' && data.models) {
        console.log(`Loaded ${data.models.length} models`)
        setModels(data.models)
        
        if (data.models.length === 0) {
          setError('No models found in SerializedModels directory')
        }
      } else {
        throw new Error(data.message || 'Failed to load models')
      }
    } catch (err) {
      console.error('Error fetching models:', err)
      setError(`Failed to load models: ${err.message}`)
      setModels([])
    } finally {
      setModelsLoading(false)
    }
  }

  /**
   * Generate image with selected model
   */
  const generateImage = async () => {
    if (!selectedModel) {
      setError('Please select a model first')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      console.log(`Generating image with model: ${selectedModel}`)
      
      const response = await fetch(GENERATE_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel,
          channels: 3
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.message || `Server error: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.status === 'success') {
        console.log('Image generated successfully')
        setImageData(data.image)
        setGenerationMode(data.mode)
      } else {
        throw new Error(data.message || 'Generation failed')
      }
    } catch (err) {
      console.error('Error generating image:', err)
      setError(err.message || 'An error occurred while generating the image')
    } finally {
      setLoading(false)
    }
  }

  /**
   * Handle model selection change
   */
  const handleModelChange = (e) => {
    setSelectedModel(e.target.value)
  }

  /**
   * Download generated image
   */
  const downloadImage = () => {
    if (!imageData) return

    const link = document.createElement('a')
    link.href = `data:image/png;base64,${imageData}`
    link.download = `generated-${selectedModel}-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-4 md:p-8">
      {/* Main container */}
      <div className="max-w-6xl mx-auto">
        {/* Content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Control Card */}
            <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700/50 rounded-2xl p-8 shadow-2xl">
              {/* Header */}
              <div className="mb-8">
                <h2 className="text-xl font-bold text-white mb-2">Generation Control</h2>
                <p className="text-sm text-slate-400">Select a model and configure generation</p>
              </div>

              {/* Model Selector */}
              <div className="mb-8">
                <label className="block text-sm font-semibold text-slate-300 mb-3">
                  Select Model
                </label>
                
                {modelsLoading ? (
                  <div className="flex items-center justify-center h-12 bg-slate-700/30 rounded-xl">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                      <span className="text-xs text-slate-400">Loading models...</span>
                    </div>
                  </div>
                ) : models.length === 0 ? (
                  <div className="bg-slate-700/30 border border-slate-600/30 rounded-xl p-4 text-center">
                    <p className="text-xs text-slate-400">No models available</p>
                    <button
                      onClick={fetchModels}
                      className="text-xs text-purple-400 hover:text-purple-300 mt-2 transition-colors"
                    >
                      Refresh
                    </button>
                  </div>
                ) : (
                  <select
                    value={selectedModel || ''}
                    onChange={handleModelChange}
                    className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 transition-all"
                  >
                    <option value="">Choose a model...</option>
                    {models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name} ({model.size_mb}MB)
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Model Info */}
              {selectedModel && models.length > 0 && (
                <div className="bg-slate-700/30 border border-slate-600/30 rounded-xl p-4 mb-8">
                  <p className="text-xs text-slate-400 mb-1">Selected Model</p>
                  <p className="text-sm font-semibold text-white truncate">
                    {selectedModel}
                  </p>
                </div>
              )}

              {/* Generate Button */}
              <button
                onClick={generateImage}
                disabled={loading || !selectedModel || modelsLoading}
                className={`w-full py-3 px-4 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${
                  loading || !selectedModel || modelsLoading
                    ? 'bg-slate-700/50 text-slate-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-purple-600 to-purple-700 text-white hover:from-purple-500 hover:to-purple-600 active:scale-95 shadow-lg hover:shadow-purple-500/50'
                }`}
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Generating...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Generate Image
                  </>
                )}
              </button>

              {/* Refresh Models Button */}
              <button
                onClick={fetchModels}
                disabled={modelsLoading}
                className="w-full mt-3 py-2 px-4 rounded-xl text-sm font-medium bg-slate-700/30 text-slate-300 hover:bg-slate-700/50 transition-colors disabled:opacity-50"
              >
                Refresh Models
              </button>
            </div>

            {/* Stats Card */}
            {selectedModel && (
              <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700/50 rounded-2xl p-6 shadow-2xl">
                <div className="space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Models Available</span>
                    <span className="font-semibold text-white">{models.length}</span>
                  </div>
                  {generationMode && (
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-slate-400">Last Generation</span>
                      <span className={`font-semibold px-2 py-1 rounded ${
                        generationMode === 'model' 
                          ? 'bg-green-500/20 text-green-300'
                          : 'bg-yellow-500/20 text-yellow-300'
                      }`}>
                        {generationMode}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - Image Display */}
          <div className="lg:col-span-2 space-y-6">
            {/* Image Container */}
            <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700/50 rounded-2xl overflow-hidden shadow-2xl">
              {loading ? (
                <div className="aspect-square flex flex-col items-center justify-center bg-slate-900/50 p-8">
                  <div className="mb-6">
                    <div className="w-16 h-16 border-4 border-slate-700 rounded-full" />
                    <div className="w-16 h-16 border-4 border-purple-500 rounded-full absolute mt-0 animate-spin border-t-transparent" />
                  </div>
                  <p className="text-slate-300 font-medium mb-2">Generating your image</p>
                  <p className="text-xs text-slate-500">This may take a few moments...</p>
                </div>
              ) : error ? (
                <div className="aspect-square flex flex-col items-center justify-center bg-red-950/30 border border-red-800/30 p-8">
                  <svg className="w-16 h-16 text-red-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4m0 4v.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h3 className="text-red-300 font-semibold mb-2">Error</h3>
                  <p className="text-sm text-red-400 text-center">{error}</p>
                  <button
                    onClick={() => setError(null)}
                    className="mt-4 px-4 py-2 bg-red-600/30 hover:bg-red-600/50 text-red-300 rounded-lg text-xs font-medium transition-colors"
                  >
                    Dismiss
                  </button>
                </div>
              ) : imageData ? (
                <div className="flex flex-col h-full">
                  <div className="flex-1 bg-slate-900/30 p-8 flex items-center justify-center">
                    <img
                      src={`data:image/png;base64,${imageData}`}
                      alt="Generated"
                      className="w-full h-full object-contain rounded-xl shadow-lg"
                      style={{ imageRendering: "pixelated" }}
                    />
                  </div>
                  <div className="border-t border-slate-700/50 p-4 flex gap-3">
                    <button
                      onClick={generateImage}
                      className="flex-1 py-2 px-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-colors active:scale-95"
                    >
                      Generate Again
                    </button>
                    <button
                      onClick={downloadImage}
                      className="flex-1 py-2 px-4 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm font-medium transition-colors active:scale-95 flex items-center justify-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Download
                    </button>
                  </div>
                </div>
              ) : (
                <div className="aspect-square flex flex-col items-center justify-center bg-slate-900/30 p-8">
                  <svg className="w-20 h-20 text-slate-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="text-slate-400 font-medium mb-1">No image generated yet</p>
                  <p className="text-sm text-slate-500 text-center">Select a model and press Generate to create an image</p>
                </div>
              )}
            </div>

            {/* Info Section */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700/50 rounded-xl p-4">
                <p className="text-xs text-slate-400 mb-1">Backend Status</p>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <p className="text-sm font-semibold text-white">Online</p>
                </div>
              </div>
              <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700/50 rounded-xl p-4">
                <p className="text-xs text-slate-400 mb-1">Models Loaded</p>
                <p className="text-sm font-semibold text-white">{models.length}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImageGenerator
