import { useState } from 'react'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <h1>MTG Chatbot</h1>
        <p>
          Welcome to the MTG Chatbot Frontend. Edit <code>src/App.jsx</code> to get started.
        </p>
      </div>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
      </div>
    </>
  )
}

export default App
