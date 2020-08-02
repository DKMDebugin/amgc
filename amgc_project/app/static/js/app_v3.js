'use strict'

//webkitURL is deprecated but nevertheless
// URL = window.URL || window.webkitURL

let gumStream 	//stream from getUserMedia()
let rec 		//Recorder.js object
let input 	    //MediaStreamAudioSourceNode that will be recorded

// shim for AudioContext when it's not avb.
let AudioContext = window.AudioContext || window.webkitAudioContext
let audioContext //audio context to help us record

let progressBar = document.querySelector('.progress')
let recordButton = document.getElementById('recordButton')
let stopButton = document.getElementById('stopButton')
let resetButton = document.getElementById('resetButton')
let sendButton = document.getElementById('sendButton')
let spinner = document.getElementById('spinner')

// Main
window.onload = function()  {
    //add events to buttons
    recordButton.addEventListener('click', startRecording)
    stopButton.addEventListener('click', stopRecording)
    resetButton.addEventListener('click', resetRecording)
    sendButton.addEventListener('click', sendRecording)
}

// Utilities
function startRecording() {
    console.log('recordButton clicked')

    let constraints = {
        audio: true,
        video: false
    }

    // disable record & send button and enable stop & reset button until getUserMedia is instantiated
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
        progressBar.style.cssText = '-webkit-animation:progressBar 30s;animation:progressBar 30s' +
            '-webkit-animation-fill-mode:both;animation-fill-mode:both'

        // set key frame for progress bar animation
        // let KeyframeRule = window.CSSRule.KEYFRAMES_RULE || window.CSSRule.WEBKIT_KEYFRAMES_RULE
        // let stylesheets = document.styleSheets
        // for (let i = 0 i < stylesheets.length i++){
        //     for (let j = 0 j < stylesheets[i].cssRules.length j++){
        //         if (stylesheets[i].cssRules[j].type == KeyframeRule) {
        //             stylesheets[i].cssRules[j].appendRule("0% { width: 0 }")
        //             stylesheets[i].cssRules[j].appendRule("100% { width: 100% }")
        //         }
        //     }
        // }

        console.log('Recording started')

    }).catch(function(err) {
        console.log('Error thrown when instantiating getUserMedia: ' + err)

        //enable & disable buttons if getUserMedia() fails
        recordButton.disabled = true
        stopButton.disabled = false
        resetButton.disabled = false
        sendButton.disabled = true
    })
}

function stopRecording(){
    console.log('pauseButton clicked rec.recording=', rec.recording )

    if (rec.recording){
        //pause
        rec.stop()

        sendButton.disabled = false
        stopButton.innerHTML='Resume'

        // stop progress bar
        progressBar.style.webkitAnimationPlayState = "paused"

    } else {
        //resume
        rec.record()
        stopButton.innerHTML='Pause'

        // resume progress bar
        progressBar.style.webkitAnimationPlayState = "running"

    }
}

function sendAudioDataAndPresentResult(blob) {
    console.log('Sending data to server side')

    spinner.style.display = 'block'

    let filename = 'audio.wav'
    let formData = new FormData()

    formData.append('audio_data', blob, filename)

    fetch('/audio', {
        method: 'POST',
        body: formData
    }).then(res => res.json())
    .then(data => {
        console.log('Present results')
        spinner.style.display = 'none'
        plotPieChart(data.prediction)
        // plotLineGraph(data.analysis)
    }).catch(err => {
        console.log(err)
    })
}

function resetRecording(){
    // clear record buffer
    rec.clear()

    stopButton.innerHTML='Pause'

    recordButton.disabled = false
    stopButton.disabled = true
    resetButton.disabled = true
    sendButton.disabled = true

    // reset progress bar
    progressBar.style.webkitAnimationName = ""

    // remove charts
    if(document.getElementById('results').children[2].children.length > 0){

        document.getElementById('pieChartDiv').children[0].remove()
        document.getElementById('lineChartDiv').children[0].remove()

    }
}

function sendRecording(){
    if (!rec.recording) {
        rec.exportWAV(sendAudioDataAndPresentResult)
    }
}

function plotPieChart(data){
    // set the dimensions and margins of the graph
    let width = 500, height = 450, margin = 40

    // The radius of the pieplot is half the width or half the height (smallest one). I subtract a bit of margin.
    let radius = Math.min(width, height) / 2 - margin

    // append the svg object to the div called 'my_dataviz'
    let svg = d3.select("#pieChartDiv")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")")

    // set the color scale
    let color = d3.scaleOrdinal()
      .domain(data)
      .range(["#B1D8B7", "#D2FBA4", "#ECF87F"])

    // Compute the position of each group on the pie:
    let pie = d3.pie()
        .sort(null) // Do not sort group by size
        .value(function (d) {
            return d.value
        })
    let data_ready = pie(d3.entries(data))

    // The arc generator
    let arc = d3.arc()
        .innerRadius(radius * 0.5)         // This is the size of the donut hole
        .outerRadius(radius * 0.8)

    // Another arc that won't be drawn. Just for labels positioning
    let outerArc = d3.arc()
        .innerRadius(radius * 0.9)
        .outerRadius(radius * 0.9)

    // Build the pie chart: Basically, each part of the pie is a path that we build using the arc function.
    svg
        .selectAll('allSlices')
        .data(data_ready)
        .enter()
        .append('path')
        .attr('d', arc)
        .attr('fill', function (d) {
            return (color(d.data.key))
        })
        .attr("stroke", "white")
        .style("stroke-width", "2px")
        .style("opacity", 0.7)

    // Add the polylines between chart and labels:
    svg
        .selectAll('allPolylines')
        .data(data_ready)
        .enter()
        .append('polyline')
        .attr("stroke", "white")
        .style("fill", "none")
        .attr("stroke-width", 1)
        .attr('points', function (d) {
            let posA = arc.centroid(d) // line insertion in the slice
            let posB = outerArc.centroid(d) // line break: we use the other arc generator that has been built only for that
            let posC = outerArc.centroid(d) // Label position = almost the same as posB
            //let midangle = d.startAngle + (d.endAngle - d.startAngle) / 2 // we need the angle to see if the X position will be at the extreme right or extreme left
            let midangle

            midangle = ((d.startAngle + (d.endAngle - d.startAngle)) * (d.index*d.index)) / 2
            if(d.index==0)
                midangle = d.startAngle + (d.endAngle - d.startAngle) / 2

            posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1) // multiply by 1 or -1 to put it on the right or on the left
            return [posA, posB, posC]
        })

    // Add the polylines between chart and labels:
    svg
        .selectAll('allLabels')
        .data(data_ready)
        .enter()
        .append('text')
        .text(function (d) {
            return d.data.key
        })
        .attr('transform', function (d) {
            let pos = outerArc.centroid(d)
            let midangle

            midangle = ((d.startAngle + (d.endAngle - d.startAngle)) * d.index) / 2
            if(d.index==0)
                midangle = d.startAngle + (d.endAngle - d.startAngle) / 2

            pos[0] = radius * 0.99 * (midangle < Math.PI ? 1 : -1)
            // console.log(pos)
            return 'translate(' + pos + ')'
        })
        .style("fill", "white")
        .style('text-anchor', function (d) {
            let midangle

            midangle = ((d.startAngle + (d.endAngle - d.startAngle)) * d.index) / 2
            if(d.index==0)
                midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
            // console.log(midangle)
            return (midangle < Math.PI ? 'start' : 'end')
        })
}

function plotLineGraph(data){
    // set the dimensions and margins of the graph
    let margin = {top: 10, right: 30, bottom: 30, left: 60},
        width = 460 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom

    // append the svg object to the body of the page
    let svg = d3.select("#lineChartDiv")
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")")

    // Add X axis
    let x = d3.scaleLinear()
      .domain([0, d3.max(data, function(d) { return +d.timestamps })])
      .range([ 0, width ])

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .style("stroke", "white")
        .call(d3.axisBottom(x))

    // svg.append("g").selectAll("text")
    // .data(data)
    // .enter()
    // .append("text")
    // .attr("transform", "translate(0," + height + ")")
    // .attr("fill", "red")
    // .text(function(d) {
    //     return "X axis"
    // });

    // Add Y axis
    let y = d3.scaleLinear()
      .domain([0, d3.max(data, function(d) { return +d.beats })])
      .range([ height, 0 ])

    svg.append("g")
        .style("stroke", "white")
        .call(d3.axisLeft(y))

    // Add the line
    svg.append("path")
      .datum(data)
      .attr("fill", "none")
        .attr("stroke", "white")
        .attr("stroke-width", 1.5)
      .attr("d", d3.line()
        .x(function(d) { return x(d.timestamps) })
        .y(function(d) { return y(d.beats) })
        )
}