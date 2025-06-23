import { useEffect, useState } from "react";
import axios from "axios";

function App() {
  const [msg, setMsg] = useState("");

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/").then(res => {
      setMsg(res.data.message);
    });
  }, []);

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Housing Price Estimator</h1>
      <p>Backend says: {msg || "Loading..."}</p>
    </div>
  );
}

export default App;