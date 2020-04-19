'use strict';

const recordAudio = () =>
  new Promise(async resolve => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks = [];

    mediaRecorder.addEventListener("dataavailable", event => {
      audioChunks.push(event.data);
    });

    const start = () => mediaRecorder.start();

    const stop = () =>
      new Promise(resolve => {
        mediaRecorder.addEventListener("stop", () => {
          const audioBlob = new Blob(audioChunks);
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);
          const play = () => audio.play();
          resolve({ audioBlob, audioUrl, play });
        });

        mediaRecorder.stop();
      });

    resolve({ start, stop });
  });

window.onload = function() {
const recorder = recordAudio();
let startBtn = document.querySelector("#startBtn")
let stopBtn = document.querySelector("#stopBtn")

const sleep = time => new Promise(resolve => setTimeout(resolve, time));

startBtn.onclick = function startRecording() {
  console.log("start!")
  recorder.then(recorder =>{
    recorder.start()
  })
}

stopBtn.onclick = function stopRecording() {
  console.log("stop!")
  recorder.then(recorder => {
    const audio = recorder.stop();
    audio.then(audio => {
      audio.play();
      let file = new File([audio.audioBlob], 'audio_file')
      // let arrayBuffer = audio.audioBlob.arrayBuffer()
      // arrayBuffer.then(arrayBuffer => {
      //   console.log(arrayBuffer)
      //   const formData = new FormData();
      //   formData.append('array_buffer', arrayBuffer)
      //
      //   fetch('/audio', {
      //     method: 'POST',
      //     body: formData
      //   }).then(res => {
      //     console.log(res.status);
      //   })
      // })
      const formData = new FormData();
      formData.append('audio_file', file)

      fetch('/audio', {
        method: 'POST',
        body: formData
      }).then(res => {
        console.log(res.status);
      })
      // const reader = new FileReader();
      // reader.readAsDataURL(audio.audioBlob);
      // reader.onload = () => {
      //   const base64AudioMessage = reader.result.split(',')[1];
      //
      //   fetch('/audio', {
      //     method: 'POST',
      //     headers: { 'Content-Type': 'application/json' },
      //     body: JSON.stringify({ message: base64AudioMessage })
      //   }).then(res => {
      //     console.log(res.status);
      //   })
      // }
    })
  })
}

// (async () => {
//   const recorder = await recordAudio();
//   recorder.start();
//   await sleep(3000);
//   const audio = await recorder.stop();
//   audio.play();
//   console.log(audio.audioBlob)
// })();

}
