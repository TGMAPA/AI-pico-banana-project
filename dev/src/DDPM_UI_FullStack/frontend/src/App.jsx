import ImageGenerator from './components/ImageGenerator'

function App() {
  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-white">
                DDPM Image Generator
              </h1>
              <p className="text-slate-400 mt-2">
                Dynamic model selection and generation
              </p>
            </div>
            <div className="text-right">
              <div className="inline-block">
                <div className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-sm text-slate-300">System Online</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main>
        <ImageGenerator />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8 items-center">
          <div className="border-t border-slate-700/50 pt-6 text-center">
            <p className="text-slate-500 text-sm">
              Advanced image generation with dynamic model selection and management
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
