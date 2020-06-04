// Main
window.onload = function() {
    //webkitURL is deprecated but nevertheless
    URL = window.URL || window.webkitURL

    let gumStream 	//stream from getUserMedia()
    let rec 		//Recorder.js object
    let input 	    //MediaStreamAudioSourceNode that will be recorded

    // shim for AudioContext when it's not avb.
    let AudioContext = window.AudioContext || window.webkitAudioContext
    let audioContext //audio context to help us record


    let recordButton = document.getElementById('recordButton')
    let stopButton = document.getElementById('stopButton')
    // let pauseButton = document.getElementById('pauseButton')

    //add events to those 2 buttons
    recordButton.addEventListener('click', startRecording)
    stopButton.addEventListener('click', stopRecording)
    // pauseButton.addEventListener('click', pauseRecording)

}

// Utilities
function startRecording() {
    console.log('recordButton clicked')

    let constraints = {
        audio: true,
        video: false
    }

    // Disable record button until getUserMedia is instantiated
    recordButton.disabled = true
    stopButton.disabled = false
    // pauseButton.disabled = false

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        console.log('getUserMedia() success, stream created, initializing Recorder.js ...')

        audioContext = new AudioContext()

        let numChannels = 2

        // update the format
        document.getElementById('formats').innerHTML='Format: ' +numChannels+ ' channel pcm @ '+audioContext.sampleRate/1000+'kHz'

        // assign to gumStream for later use
        gumStream = stream

        // use the stream
        input = audioContext.createMediaStreamSource(stream)

        // create Recorder.js object and configure to record mono sound via numChannels to minimize file size
        rec = new Recorder(input, {numChannels: numChannels})

        //start the recording process
        rec.record()

        console.log('Recording started')

    }).catch(function(err) {
        //enable the record button if getUserMedia() fails
        recordButton.disabled = false
        stopButton.disabled = true
        // pauseButton.disabled = true
    })
}

function stopRecording(){
    console.log('pauseButton clicked rec.recording=',rec.recording )

    if (rec.recording){
        //pause
        rec.stop()
        // stopButton.innerHTML='Resume'
        rec.exportWAV(sendAudioData)
    }else{
        //resume
        rec.record()
        // stopButton.innerHTML='Pause'

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