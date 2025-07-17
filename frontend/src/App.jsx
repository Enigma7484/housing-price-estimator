import { useState } from "react";
import axios from "axios";
import { Box, Button, Container, Typography, TextField, Paper, Grid } from "@mui/material";

export default function App() {
  const [form, setForm]   = useState({ bed:3, bath:2, sqft:1500, n_citi:"182" });
  const [price, setPrice] = useState(null);

  const handle = e => setForm({...form, [e.target.name]:e.target.value});

  const submit = async () => {
    const res = await axios.post("http://localhost:8000/predict", form);
    setPrice(res.data.predicted_price);
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 6 }}>
      <Paper elevation={4} sx={{ p: 4, borderRadius: 3 }}>
        <Typography variant="h4" align="center" gutterBottom>
          🏠 Housing Price Estimator
        </Typography>
        <Box component="form" noValidate autoComplete="off">
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                label="Bedrooms"
                name="bed"
                type="number"
                value={form.bed}
                onChange={handle}
                fullWidth
                variant="outlined"
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                label="Bathrooms"
                name="bath"
                type="number"
                value={form.bath}
                onChange={handle}
                fullWidth
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Square Feet"
                name="sqft"
                type="number"
                value={form.sqft}
                onChange={handle}
                fullWidth
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="City Code"
                name="n_citi"
                value={form.n_citi}
                onChange={handle}
                fullWidth
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                onClick={submit}
                fullWidth
                size="large"
                sx={{ mt: 2 }}
              >
                Estimate
              </Button>
            </Grid>
          </Grid>
        </Box>
        {price && (
          <Typography
            variant="h5"
            align="center"
            color="success.main"
            sx={{ mt: 4 }}
          >
            ≈ ${price.toLocaleString()}
          </Typography>
        )}
      </Paper>
    </Container>
  );
}