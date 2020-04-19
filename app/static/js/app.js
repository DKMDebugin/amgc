'use strict';
let fs = require('fs')
let Ffmpeg = require('ffmpeg')

// Main
window.onload = function() {
  const recorder = recordAudio();
  let startBtn = document.querySelector("#startBtn")
  let stopBtn = document.querySelector("#stopBtn")

  const sleep = time => new Promise(resolve => setTimeout(resolve, time))

  startBtn.onclick = startRecording(recorder)

  stopBtn.onclick = stopRecording(recorder)
  // function stopRecording() {
//   console.log("stop!")
//   recorder.then(recorder => {
//     const audio = recorder.stop();
//     audio.then(audio => {
//       audio.play();
//       let file = new File([audio.audioBlob], 'audio_file')
//       // let arrayBuffer = audio.audioBlob.arrayBuffer()
//       // arrayBuffer.then(arrayBuffer => {
//       //   console.log(arrayBuffer)
//       //   const formData = new FormData();
//       //   formData.append('array_buffer', arrayBuffer)
//       //
//       //   fetch('/audio', {
//       //     method: 'POST',
//       //     body: formData
//       //   }).then(res => {
//       //     console.log(res.status);
//       //   })
//       // })
//       const formData = new FormData();
//       formData.append('audio_file', file)
//
//       fetch('/audio', {
//         method: 'POST',
//         body: formData
//       }).then(res => {
//         console.log(res.status);
//       })
//       // const reader = new FileReader();
//       // reader.readAsDataURL(audio.audioBlob);
//       // reader.onload = () => {
//       //   const base64AudioMessage = reader.result.split(',')[1];
//       //
//       //   fetch('/audio', {
//       //     method: 'POST',
//       //     headers: { 'Content-Type': 'application/json' },
//       //     body: JSON.stringify({ message: base64AudioMessage })
//       //   }).then(res => {
//       //     console.log(res.status);
//       //   })
//       // }
//     })
//   })
// }
//
// // (async () => {
// //   const recorder = await recordAudio();
//   recorder.start();
//   await sleep(3000);
//   const audio = await recorder.stop();
//   audio.play();
//   console.log(audio.audioBlob)
// })();

}

// Utility functions
// Record audio live and return a blod object, blob url, play function
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

// Starts recording live audio
function startRecording(recorder) {
  return () => {
    console.log("start!")
    recorder.then(recorder =>{
      recorder.start()
    })
  }
}

// Stops recording then calls convertToMp3() and sendAudioToServerSide()
function stopRecording(recorder) {
  return () => {
    console.log("stop!")
    recorder.then(recorder => {
      const audio = recorder.stop();
      audio.then(audio => {
        let filename = 'audioBlob.txt'
        audio.play();
        // console.log(audio.audioUrl)
        saveBlobAsFile(audio.audioBlob, filename)
        convertToMp3(filename)

        // sendAudioToServerSide()

      })
    })
  }
}

function saveBlobAsFile(blob, filename) {
  fs.writeFile(filename, blob, function (err) {
    if (err) throw err;
    console.log('Saved!');
  })
}
// let saveBlobAsFile = (function () {
//     var a = document.createElement("a");
//     document.body.appendChild(a);
//     a.style = "display: none";
//     return (url, filename) => {
//         a.href = url;
//         a.download = filename;
//         a.click();
//         window.URL.revokeObjectURL(url);
//         console.log('Done!')
//     };
// }());

function convertToMp3(pathToBlob) {
  try {
    console.log(pathToBlob)
    let process = new Ffmpeg(pathToBlob)
    process.then((audio) => {
      console.log(audio)
      audio.fnExtractSoundToMp3('audio.mp3', (err, file) => {
        if (!err){
          console.log('Audio file: ' + file)
        }
      })
    }, (err) => {
      console.log('Error: ' + err)
    })
	} catch (err) {
  	console.log(err.code)
  	console.log(err.msg)
  }
}

// Sends mp3 file to server side
function sendAudioToServerSide(pathToMp3) {
  let audioMp3 = fetch(pathToMp3)
  const formData = new FormData();
  formData.append('audio_file', audioMp3)

  fetch('/audio', {
    method: 'POST',
    body: formData
  }).then(res => {
    console.log(res.status);
  })
}
