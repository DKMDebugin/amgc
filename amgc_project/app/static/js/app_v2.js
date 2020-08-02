'use strict';

//webkitURL is deprecated but nevertheless
// URL = window.URL || window.webkitURL

let gumStream 	//stream from getUserMedia()
let rec 		//Recorder.js object
let input 	    //MediaStreamAudioSourceNode that will be recorded

// shim for AudioContext when it's not avb.
let AudioContext = window.AudioContext || window.webkitAudioContext
let audioContext //audio context to help us record

let progressBar = document.querySelector('.meter')
let recordButton = document.getElementById('recordButton')
let stopButton = document.getElementById('stopButton')
let resetButton = document.getElementById('resetButton')
let sendButton = document.getElementById('sendButton')

// Main
window.onload = function() {

    //add events to those 2 buttons
    recordButton.addEventListener('click', startRecording)
    stopButton.addEventListener('click', stopRecording)
    resetButton.addEventListener('click', resetRecording)
    sendButton.addEventListener('click', sendRecording)


    progressBar.style.cssText = 'animation-play-state:paused;webkitAnimationPlayState:paused;'

}

// Utilities

function startRecording() {
    console.log('recordButton clicked')

    let constraints = {
        audio: true,
        video: false
    }

    // Disable record & send button and enable stop & reset button  until getUserMedia is instantiated
    recordButton.disabled = true
    stopButton.disabled = false
    resetButton.disabled = false
    sendButton.disabled = true


    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        console.log('getUserMedia() success, stream created, initializing Recorder.js ...')

        audioContext = new AudioContext()

        let numChannels = 2

        // update the format
        let formats = document.getElementById('formats')
        formats.innerHTML='Format: ' +numChannels+ ' channel pcm @ '+audioContext.sampleRate/1000+'kHz'

        // assign to gumStream for later use
        gumStream = stream

        // use the stream
        input = audioContext.createMediaStreamSource(stream)

        // create Recorder.js object and configure to record mono sound via numChannels to minimize file size
        rec = new Recorder(input, {numChannels: numChannels})

        //start the recording process
        rec.record()

        // start progress bar
        progressBar.style.cssText = 'animation-play-state:running;webkitAnimationPlayState:running;'

        console.log('Recording started')

    }).catch(function(err) {
        console.log('Error thrown when instantiating getUserMedia: ' + err)

        //enable the record button if getUserMedia() fails
        // recordButton.disabled = false
        // stopButton.disabled = true
        // resetButton.disabled = true
        // sendButton.disabled = true
        recordButton.disabled = true
        stopButton.disabled = false
        resetButton.disabled = false
        sendButton.disabled = true
    })
}

function stopRecording(){
    console.log('pauseButton clicked rec.recording=', rec.recording )

    // recordButton.disabled = false
    // stopButton.disabled = true

    if (rec.recording){
        //pause
        rec.stop()

        sendButton.disabled = false
        stopButton.innerHTML='Resume'

        progressBar.style.cssText = 'animation-play-state:paused;webkitAnimationPlayState:paused;'

    } else {
        //resume
        rec.record()
        stopButton.innerHTML='Pause'

    }
}

function sendAudioData(blob) {
    console.log('Sending data to server side')

    let filename = 'audio.wav'
    let formData = new FormData();

    formData.append('audio_data', blob, filename)

    fetch('/audio', {
        method: 'POST',
        body: formData
    }).then(res => res.json())
    .then(res => {
        console.log(res)
        // console.log(res.data)
    }).catch(err => {
        console.log(err)
    })
}

function resetRecording(){
    rec.clear()
    // location.reload()
    recordButton.disabled = false
    stopButton.disabled = true
    resetButton.disabled = true
    sendButton.disabled = true

    // var el = document.getElementById('animated');
    progressBar.style.animation = 'none'
    // progressBar.style.webkitAnimation = 'none'
    progressBar.offsetHeight; /* trigger reflow */
    progressBar.style.animation = null;
}

function sendRecording(){
    if (!rec.recording) {
        rec.exportWAV(sendAudioData)
    }
}