const express = require('express');
const multer = require('multer');
const kafka = require('kafka-node');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

const client = new kafka.KafkaClient({ kafkaHost: process.env.KAFKA_BOOTSTRAP_SERVERS });
const producer = new kafka.Producer(client);
const inferenceResults = new Map();

app.use(express.static('public'));
app.use(express.urlencoded({ extended: true }));

app.post('/submit', upload.single('image'), (req, res) => {
  const { model, device } = req.body;
  const imagePath = req.file.path;
  const jobId = `job_${Date.now()}`; // Generate job_id

  fs.readFile(imagePath, { encoding: 'base64' }, (err, base64Image) => {
    if (err) {
      return res.status(500).send(`Error reading uploaded image: ${err.message}`);
    }

    const payloads = [{
      topic: 'inference_jobs',
      messages: JSON.stringify({
        job_id: jobId, // include job_id in Kafka message
        model,
        device,
        image: base64Image,
        filename: req.file.originalname,
        mimetype: req.file.mimetype,
        timestamp: Date.now()
      })
    }];

    producer.send(payloads, (err) => {
      fs.unlink(imagePath, () => {}); // Clean up

      if (err) {
        return res.status(500).send(`Kafka error: ${err.message || err}`);
      }

      // Send job_id back so frontend can redirect
      res.json({ job_id: jobId });
    });
  });
});

// GET /result/:jobId - polling endpoint to retrieve inference result
app.get('/result/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const result = inferenceResults.get(jobId);

  if (result) {
    res.json(result);
  } else {
    res.status(202).send({ status: 'pending' }); // 202 = Accepted but not ready
  }
});

async function waitForTopic(client, topic, retries = 10) {
  for (let i = 0; i < retries; i++) {
    const metadata = await new Promise((resolve, reject) => {
      client.loadMetadataForTopics([topic], (err, results) => {
        if (err) return reject(err);
        resolve(results);
      });
    });

    const topicsMetadata = metadata[1].metadata;
    if (topicsMetadata && topicsMetadata[topic]) {
      console.log(`Topic exists.`);
      return;
    }
    await new Promise(r => setTimeout(r, 3000));
  }
  throw new Error(`Topic not found after retries`);
}

(async () => {
  try {
    await waitForTopic(client, 'inference_results');
    const consumer = new kafka.Consumer(client, [{ topic: 'inference_results' }], { autoCommit: true });
    
    consumer.on('message', (message) => {
      const result = JSON.parse(message.value);
      const jobId = result.job_id;
      if (jobId) {
        inferenceResults.set(jobId, result);
      }
    });

  } catch (err) {
    console.error('Error: waiting for inference results', err);
    process.exit(1);
  }
})();

// Start server
app.listen(3001, () => console.log('users-api running on port 3001'));
