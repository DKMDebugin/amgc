(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
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

},{"ffmpeg":2,"fs":27}],2:[function(require,module,exports){
module.exports = require('./lib/ffmpeg');
},{"./lib/ffmpeg":5}],3:[function(require,module,exports){
/**
 * Basic configuration
 */
module.exports = function () {
	this.encoding	= 'utf8';
	this.timeout	= 0;
	this.maxBuffer	= 200 * 1024
}
},{}],4:[function(require,module,exports){
var util = require('util');

// Error list with code and message
var list = {
	'empty_input_filepath'							: { 'code' : 100, 'msg' : 'The input file path can not be empty' }
  , 'input_filepath_must_be_string'					: { 'code' : 101, 'msg' : 'The input file path must be a string' }
  , 'invalid_option_name'							: { 'code' : 102, 'msg' : 'The option "%s" is invalid. Check the list of available options' }
  , 'fileinput_not_exist'							: { 'code' : 103, 'msg' : 'The input file does not exist' }
  , 'format_not_supported'							: { 'code' : 104, 'msg' : 'The format "$s" is not supported by the version of ffmpeg' }
  , 'audio_channel_is_invalid'						: { 'code' : 105, 'msg' : 'The audio channel "$s" is not valid' }
  , 'mkdir'											: { 'code' : 106, 'msg' : 'Error occurred during creation folder: $s' }
  , 'extract_frame_invalid_everyN_options'			: { 'code' : 107, 'msg' : 'You can specify only one option between everyNFrames and everyNSeconds' }
  , 'invalid_watermark'								: { 'code' : 108, 'msg' : 'The watermark "%s" does not exists' }
  , 'invalid_watermark_position'					: { 'code' : 109, 'msg' : 'Invalid watermark position "%s"' }
  , 'size_format'									: { 'code' : 110, 'msg' : 'The format "%s" not supported by the function "setSize"' }
  , 'resolution_square_not_defined'					: { 'code' : 111, 'msg' : 'The resolution for pixel aspect ratio is not defined' }
  , 'command_already_exists'						: { 'code' : 112, 'msg' : 'The command "%s" already exists' }
  , 'codec_not_supported'							: { 'code' : 113, 'msg' : 'The codec "$s" is not supported by the version of ffmpeg' }
}

/**
 * Return the error by the codename
 */
var renderError = function (codeName) {
	// Get the error object by the codename
	var params = [list[codeName].msg];
	// Get the possible arguments
	if (arguments.length > 1)
		params = params.concat(Array.prototype.slice.call(arguments, 1));
	// Call the function for replace the letter '%s' with the found arguments
	return { 'code' : list[codeName].code, 'msg' : util.format.apply(this, params) };
}

module.exports.list = list;
module.exports.renderError = renderError;
},{"util":32}],5:[function(require,module,exports){
var when		= require('when')
  , fs			= require('fs');

var errors		= require('./errors')
  , utils		= require('./utils')
  , configs		= require('./configs')
  , video		= require('./video');

var ffmpeg = function (/* inputFilepath, settings, callback */) {

	/**
	 * Retrieve the list of the codec supported by the ffmpeg software
	 */
	var _ffmpegInfoConfiguration = function (settings) {
		// New 'promise' instance 
		var deferred = when.defer();
		// Instance the new arrays for the format
		var format = { modules : new Array(), encode : new Array(), decode : new Array() };
		// Make the call to retrieve information about the ffmpeg
		utils.exec(['ffmpeg','-formats','2>&1'], settings, function (error, stdout, stderr) {
			// Get the list of modules
			var configuration = /configuration:(.*)/.exec(stdout);
			// Check if exists the configuration
			if (configuration) {
				// Get the list of modules
				var modules = configuration[1].match(/--enable-([a-zA-Z0-9\-]+)/g);
				// Scan all modules
				for (var indexModule in modules) {
					// Add module to the list
					format.modules.push(/--enable-([a-zA-Z0-9\-]+)/.exec(modules[indexModule])[1]);
				}
			}
			// Get the codec list
			var codecList = stdout.match(/ (DE|D|E) (.*) {1,} (.*)/g);
			// Scan all codec
			for (var i in codecList) {
				// Get the match value
				var match = / (DE|D|E) (.*) {1,} (.*)/.exec(codecList[i]);
				// Check if match is valid
				if (match) {
					// Get the value from the match
					var scope = match[1].replace(/\s/g,'')
					  , extension = match[2].replace(/\s/g,'');
					// Check which scope is best suited
					if (scope == 'D' || scope == 'DE')
						format.decode.push(extension);
					if (scope == 'E' || scope == 'DE')
						format.encode.push(extension);
				}
			}
			// Returns the list of supported formats
			deferred.resolve(format);
		});
		// Return 'promise' instance 
		return deferred.promise;
	}
	
	/**
	 * Get the video info
	 */
	var _videoInfo = function (fileInput, settings) {
		// New 'promise' instance 
		var deferred = when.defer();
		// Make the call to retrieve information about the ffmpeg
		utils.exec(['ffmpeg','-i',fileInput,'2>&1'], settings, function (error, stdout, stderr) {
			// Perse output for retrieve the file info
			var filename		= /from \'(.*)\'/.exec(stdout) || []
			  , title			= /(INAM|title)\s+:\s(.+)/.exec(stdout) || []
			  , artist			= /artist\s+:\s(.+)/.exec(stdout) || []
			  , album			= /album\s+:\s(.+)/.exec(stdout) || []
			  , track			= /track\s+:\s(.+)/.exec(stdout) || []
			  , date			= /date\s+:\s(.+)/.exec(stdout) || []
			  , is_synched		= (/start: 0.000000/.exec(stdout) !== null)
			  , duration		= /Duration: (([0-9]+):([0-9]{2}):([0-9]{2}).([0-9]+))/.exec(stdout) || []
			  
			  , container		= /Input #0, ([a-zA-Z0-9]+),/.exec(stdout) || []
			  , video_bitrate	= /bitrate: ([0-9]+) kb\/s/.exec(stdout) || []
			  , video_stream	= /Stream #([0-9\.]+)([a-z0-9\(\)\[\]]*)[:] Video/.exec(stdout) || []
			  , video_codec		= /Video: ([\w]+)/.exec(stdout) || []
			  , resolution		= /(([0-9]{2,5})x([0-9]{2,5}))/.exec(stdout) || []
			  , pixel			= /[SP]AR ([0-9\:]+)/.exec(stdout) || []
			  , aspect			= /DAR ([0-9\:]+)/.exec(stdout) || []
			  , fps				= /([0-9\.]+) (fps|tb\(r\))/.exec(stdout) || []
			  
			  , audio_stream	= /Stream #([0-9\.]+)([a-z0-9\(\)\[\]]*)[:] Audio/.exec(stdout) || []
			  , audio_codec		= /Audio: ([\w]+)/.exec(stdout) || []
			  , sample_rate		= /([0-9]+) Hz/i.exec(stdout) || []
			  , channels		= /Audio:.* (stereo|mono)/.exec(stdout) || []
			  , audio_bitrate	= /Audio:.* ([0-9]+) kb\/s/.exec(stdout) || []
			  , rotate			= /rotate[\s]+:[\s]([\d]{2,3})/.exec(stdout) || [];
			// Build return object
			var ret = { 
				filename		: filename[1] || ''
			  , title			: title[2] || ''
			  , artist			: artist[1] || ''
			  , album			: album[1] || ''
			  , track			: track[1] || ''
			  , date			: date[1] || ''
			  , synched			: is_synched
			  , duration		: {
					raw		: duration[1] || ''
				  , seconds	: duration[1] ? utils.durationToSeconds(duration[1]) : 0
				}
			  , video			: {
					container			: container[1] || ''
				  , bitrate				: (video_bitrate.length > 1) ? parseInt(video_bitrate[1], 10) : 0
				  , stream				: video_stream.length > 1 ? parseFloat(video_stream[1]) : 0.0
				  , codec				: video_codec[1] || ''
				  , resolution			: {
						w : resolution.length > 2 ? parseInt(resolution[2], 10) : 0
					  , h : resolution.length > 3 ? parseInt(resolution[3], 10) : 0
					}
				  , resolutionSquare	: {}
				  , aspect				: {}
				  , rotate				: rotate.length > 1 ? parseInt(rotate[1], 10) : 0
				  , fps					: fps.length > 1 ? parseFloat(fps[1]) : 0.0
				}
			  , audio			: {
					codec				: audio_codec[1] || ''
				  , bitrate				: audio_bitrate[1] || ''
				  , sample_rate			: sample_rate.length > 1 ? parseInt(sample_rate[1], 10) : 0
				  , stream				: audio_stream.length > 1 ? parseFloat(audio_stream[1]) : 0.0
				  , channels			: {
						raw		: channels[1] || ''
					  , value	: (channels.length > 0) ? ({ stereo : 2, mono : 1 }[channels[1]] || 0) : ''
					}
				}
			};
			// Check if exist aspect ratio
			if (aspect.length > 0) {
				var aspectValue = aspect[1].split(":");
				ret.video.aspect.x		= parseInt(aspectValue[0], 10);
				ret.video.aspect.y		= parseInt(aspectValue[1], 10);
				ret.video.aspect.string = aspect[1];
				ret.video.aspect.value	= parseFloat((ret.video.aspect.x / ret.video.aspect.y));
			} else {
				// If exists horizontal resolution then calculate aspect ratio
				if(ret.video.resolution.w > 0) {
					var gcdValue = utils.gcd(ret.video.resolution.w, ret.video.resolution.h);
					// Calculate aspect ratio
					ret.video.aspect.x		= ret.video.resolution.w / gcdValue;
					ret.video.aspect.y		= ret.video.resolution.h / gcdValue;
					ret.video.aspect.string = ret.video.aspect.x + ':' + ret.video.aspect.y;
					ret.video.aspect.value	= parseFloat((ret.video.aspect.x / ret.video.aspect.y));
				}
			}
			// Save pixel ratio for output size calculation
			if (pixel.length > 0) {
				ret.video.pixelString = pixel[1];
				var pixelValue = pixel[1].split(":");
				ret.video.pixel = parseFloat((parseInt(pixelValue[0], 10) / parseInt(pixelValue[1], 10)));
			} else {
				if (ret.video.resolution.w !== 0) {
					ret.video.pixelString = '1:1';
					ret.video.pixel = 1;
				} else {
					ret.video.pixelString = '';
					ret.video.pixel = 0.0;
				}
			}
			// Correct video.resolution when pixel aspectratio is not 1
			if (ret.video.pixel !== 1 || ret.video.pixel !== 0) {
				if( ret.video.pixel > 1 ) {
					ret.video.resolutionSquare.w = parseInt(ret.video.resolution.w * ret.video.pixel, 10);
					ret.video.resolutionSquare.h = ret.video.resolution.h;
				} else {
					ret.video.resolutionSquare.w = ret.video.resolution.w;
					ret.video.resolutionSquare.h = parseInt(ret.video.resolution.h / ret.video.pixel, 10);
				}
			}
			// Returns the list of supported formats
			deferred.resolve(ret);
		});
		// Return 'promise' instance 
		return deferred.promise;
	}
	
	/**
	 * Get the info about ffmpeg's codec and about file
	 */
	var _getInformation = function (fileInput, settings) {
		var deferreds = [];
		// Add promise
		deferreds.push(_ffmpegInfoConfiguration(settings));
		deferreds.push(_videoInfo(fileInput, settings));
		// Return defer
		return when.all(deferreds);
	}

	var __constructor = function (args) {
		// Check if exist at least one option
		if (args.length == 0 || args[0] == undefined)
			throw errors.renderError('empty_input_filepath');
		// Check if first argument is a string
		if (typeof args[0] != 'string')
			throw errors.renderError('input_filepath_must_be_string');
		// Get the input filepath
		var inputFilepath = args[0];
		// Check if file exist
		if (!fs.existsSync(inputFilepath))
			throw errors.renderError('fileinput_not_exist');
		
		// New instance of the base configuration
		var settings = new configs();
		// Callback to call
		var callback = null;
		
		// Scan all arguments
		for (var i = 1; i < args.length; i++) {
			// Check the type of variable
			switch (typeof args[i]) {
				case 'object' :
					utils.mergeObject(settings, args[i]);
					break;
				case 'function' :
					callback = args[i];
					break;
			}
		}
		
		// Building the value for return value. Check if the callback is not a function. In this case will created a new instance of the deferred class
		var deferred = typeof callback != 'function' ? when.defer() : { promise : null };
		
		when(_getInformation(inputFilepath, settings), function (data) {
			// Check if the callback is a function
			if (typeof callback == 'function') {
				// Call the callback function e return the new instance of 'video' class
				callback(null, new video(inputFilepath, settings, data[0], data[1]));
			} else {
				// Positive response
				deferred.resolve(new video(inputFilepath, settings, data[0], data[1]));
			}
		}, function (error) {
			// Check if the callback is a function
			if (typeof callback == 'function') {
				// Call the callback function e return the error found
				callback(error, null);
			} else {
				// Negative response
				deferred.reject(error);
			}
		});
		
		// Return a possible promise instance
		return deferred.promise;
	}

	return __constructor.call(this, arguments);
};

module.exports = ffmpeg;
},{"./configs":3,"./errors":4,"./utils":7,"./video":8,"fs":27,"when":26}],6:[function(require,module,exports){
module.exports.size = {
	'SQCIF'		: '128x96'
  ,	'QCIF'		: '176x144'
  ,	'CIF'		: '352x288'
  ,	'4CIF'		: '704x576'
  ,	'QQVGA'		: '160x120'
  ,	'QVGA'		: '320x240'
  ,	'VGA'		: '640x480'
  ,	'SVGA'		: '800x600'
  ,	'XGA'		: '1024x768'
  ,	'UXGA'		: '1600x1200'
  ,	'QXGA'		: '2048x1536'
  ,	'SXGA'		: '1280x1024'
  ,	'QSXGA'		: '2560x2048'
  ,	'HSXGA'		: '5120x4096'
  ,	'WVGA'		: '852x480'
  ,	'WXGA'		: '1366x768'
  ,	'WSXGA'		: '1600x1024'
  ,	'WUXGA'		: '1920x1200'
  ,	'WOXGA'		: '2560x1600'
  ,	'WQSXGA'	: '3200x2048'
  ,	'WQUXGA'	: '3840x2400'
  ,	'WHSXGA'	: '6400x4096'
  ,	'WHUXGA'	: '7680x4800'
  ,	'CGA'		: '320x200'
  ,	'EGA'		: '640x350'
  ,	'HD480'		: '852x480'
  ,	'HD720'		: '1280x720'
  ,	'HD1080'	: '1920x1080'
}

module.exports.ratio = {
	'4:3'		: 1.33
  , '3:2'		: 1.5
  , '14:9'		: 1.56
  , '16:9'		: 1.78
  , '21:9'		: 2.33
}

module.exports.audio_channel = {
	'mono'		: 1
  ,	'stereo'	: 2
}
},{}],7:[function(require,module,exports){
var exec	= require('child_process').exec
  , fs		= require('fs')
  , path	= require('path');

var errors	= require('./errors');

/**
 * Exec the list of commands and call the callback function at the end of the process
 */
module.exports.exec = function (commands, settings, callback) {
	// Create final command line
	var finalCommand = commands.join(" ");
	// Create the timeoutId for stop the timeout at the end the process
	var timeoutID = null;
	// Exec the command
	var process = exec(finalCommand, settings, function (error, stdout, stderr) {
		// Clear timeout if 'timeoutID' are setted
		if (timeoutID !== null) clearTimeout(timeoutID);
		// Call the callback function
		callback(error, stdout, stderr);
	});
	// Verify if the timeout are setting
	if (settings.timeout > 0) {
		// Set the timeout
		timeoutID = setTimeout(function () {
			process.kill();
		}, 100);		
	}
}

/**
 * Check if object is empty
 */
module.exports.isEmptyObj = function (obj) {
	// Scan all properties
    for(var prop in obj)
		// Check if obj has a property
        if(obj.hasOwnProperty(prop))
			// The object is not empty
            return false;
	// The object is empty
    return true;
}

/**
 * Merge obj1 into obj
 */
module.exports.mergeObject = function (obj, obj1) {
	// Check if there are options set
	if (!module.exports.isEmptyObj(obj1)) {
		// Scan all settings
		for (var key in obj1) {
			// Check if the option is valid
			if (!obj.hasOwnProperty(key))
				throw errors.renderError('invalid_option_name', key);
			// Set new option value
			obj[key] = obj1[key];
		}
	}
}

/**
 * Calculate the duration in seconds from the string retrieved by the ffmpeg info
 */
module.exports.durationToSeconds = function(duration) {
	var parts = duration.substr(0,8).split(':');
	return parseInt(parts[0], 10) * 3600 + parseInt(parts[1], 10) * 60 + parseInt(parts[2], 10);
};

/**
 * Calculate the greatest common divisor
 */
module.exports.gcd = function (a, b) { 
	if (b === 0) return a;
	return module.exports.gcd(b, a % b);
}

/**
 * Offers functionality similar to mkdir -p
 */
module.exports.mkdir = function (dirpath, mode, callback, position) {
	// Split all directories
    var parts = path.normalize(dirpath).split('/');
	// If the first part is empty then remove this part
	if (parts[0] == "") 
		parts = parts.slice(1);
	
	// Set the initial configuration
    mode = mode || 0777;
    position = position || 0;
	
	// Check se current position is greater than the list of folders
	if (position > parts.length) {
		// If isset the callback then it will be invoked
		if (callback) 
			callback();
		// Exit and return a positive value
		return true;
	}

	// Build the directory path
	var directory = (dirpath.charAt(0) == '/' ? '/' : '') + parts.slice(0, position + 1).join('/');

	// Check if directory exists
	if (fs.existsSync(directory)) {
		module.exports.mkdir(dirpath, mode, callback, position + 1);
	} else {
		if (fs.mkdirSync(directory, mode)) {
			// If isset the callback then it will be invoked
			if (callback) 
				callback(errors.renderError('mkdir', directory));
			// Send the new exception
			throw errors.renderError('mkdir', directory);
		} else {
			module.exports.mkdir(dirpath, mode, callback, position + 1);
		}
	}
}

/**
 * Check if a value is present inside an array
 */
module.exports.in_array = function (value, array) {
	// Scan all element
	for (var i in array)
		// Check if value exists
		if (array[i] == value)
			// Return the position of value
			return i;
	// The value not exists
	return false;
}
},{"./errors":4,"child_process":27,"fs":27,"path":28}],8:[function(require,module,exports){
var fs			= require('fs')
  , path		= require('path')
  , when		= require('when');

var errors		= require('./errors')
  , presets		= require('./presets')
  , utils		= require('./utils');

module.exports = function (filePath, settings, infoConfiguration, infoFile) {
	
	// Public info about file and ffmpeg configuration
	this.file_path				= filePath;
	this.info_configuration		= infoConfiguration;
	this.metadata				= infoFile;
	
	// Commands for building the ffmpeg string conversion
	var commands		= new Array()
	  , inputs			= new Array()
	  , filtersComlpex	= new Array()
	  , output			= null;
	
	// List of options generated from setting functions
	var options			= new Object();
	
	/*****************************************/
	/* FUNCTION FOR FILL THE COMMANDS OBJECT */
	/*****************************************/
	
	/**
	 * Add a command to be bundled into the ffmpeg command call
	 */
	this.addCommand = function (command, argument) {
		// Check if exists the current command
		if (utils.in_array(command, commands) === false) {
			// Add the new command
			commands.push(command);
			// Add the argument to new command
			if (argument != undefined)
				commands.push(argument);
		} else 
			throw errors.renderError('command_already_exists', command);
	}
	
	/**
	 * Add an input stream
	 */
	this.addInput = function (argument) {
		inputs.push(argument);
	}
	
	/**
	 * Add a filter complex
	 */
	this.addFilterComplex = function (argument) {
		filtersComlpex.push(argument);
	}
	
	/**
	 * Set the output path
	 */
	var setOutput = function (path) {
		output = path;
	}
	
	/*********************/
	/* SETTING FUNCTIONS */
	/*********************/
	
	/**
	 * Disables audio encoding
	 */
	this.setDisableAudio = function () {
		if (options.audio == undefined)
			options.audio = new Object();
		// Set the new option
		options.audio.disabled = true;
		return this;
	}

	/**
	 * Disables video encoding
	 */
	this.setDisableVideo = function () {
		if (options.video == undefined)
			options.video = new Object();
		// Set the new option
		options.video.disabled = true;
		return this;
	}
	
	/**
	 * Sets the new video format
	 */
	this.setVideoFormat = function (format) {
		// Check if the format is supported by ffmpeg version
		if (this.info_configuration.encode.indexOf(format) != -1) {
			if (options.video == undefined)
				options.video = new Object();
			// Set the new option
			options.video.format = format;
			return this;
		} else 
			throw errors.renderError('format_not_supported', format);
	}
	
	/**
	 * Sets the new audio codec
	 */
	this.setVideoCodec = function (codec) {
		// Check if the codec is supported by ffmpeg version
		if (this.info_configuration.encode.indexOf(codec) != -1) {
			if (options.video == undefined)
				options.video = new Object();
			// Set the new option
			options.video.codec = codec;
			return this;
		} else 
			throw errors.renderError('codec_not_supported', codec);
	}
	
	/**
	 * Sets the video bitrate
	 */
	this.setVideoBitRate = function (bitrate) {
		if (options.video == undefined)
			options.video = new Object();
		// Set the new option
		options.video.bitrate = bitrate;
		return this;
	}
	
	/**
	 * Sets the framerate of the video
	 */
	this.setVideoFrameRate = function (framerate) {
		if (options.video == undefined)
			options.video = new Object();
		// Set the new option
		options.video.framerate = framerate;
		return this;		
	}
	
	/**
	 * Sets the start time
	 */
	this.setVideoStartTime = function (time) {
		if (options.video == undefined)
			options.video = new Object();
		
		// Check if time is a string that contain: hours, minutes and seconds
		if (isNaN(time) && /([0-9]+):([0-9]{2}):([0-9]{2})/.exec(time)) {
			time = utils.durationToSeconds(time);			
		} else if (!isNaN(time) && parseInt(time) == time) {
			time = parseInt(time, 10);			
		} else {
			time = 0;			
		}

		// Set the new option
		options.video.startTime = time;
		return this;
	}
	
	/**
	 * Sets the duration
	 */
	this.setVideoDuration = function (duration) {
		if (options.video == undefined)
			options.video = new Object();
		
		// Check if duration is a string that contain: hours, minutes and seconds
		if (isNaN(duration) && /([0-9]+):([0-9]{2}):([0-9]{2})/.exec(duration)) {
			duration = utils.durationToSeconds(duration);
		} else if (!isNaN(duration) && parseInt(duration) == duration) {
			duration = parseInt(duration, 10);			
		} else {
			duration = 0;
		}

		// Set the new option
		options.video.duration = duration;
		return this;
	}
	
	/**
	 * Sets the new aspetc ratio
	 */
	this.setVideoAspectRatio = function (aspect) {
		// Check if aspect is a string
		if (isNaN(aspect)) {
			// Check if aspet is string xx:xx
			if (/([0-9]+):([0-9]+)/.exec(aspect)) {
				var check = /([0-9]+):([0-9]+)/.exec(aspect);
				aspect = parseFloat((check[1] / check[2]));
			} else {
				aspect = this.metadata.video.aspect.value;
			}
		}
		
		if (options.video == undefined)
			options.video = new Object();
		// Set the new option
		options.video.aspect = aspect;
		return this;
	}
	
	/**
	 * Set the size of the video
	 */
	this.setVideoSize = function (size, keepPixelAspectRatio, keepAspectRatio, paddingColor) {
		if (options.video == undefined)
			options.video = new Object();
		// Set the new option
		options.video.size = size;
		options.video.keepPixelAspectRatio = keepPixelAspectRatio;
		options.video.keepAspectRatio = keepAspectRatio;
		options.video.paddingColor = paddingColor;
		return this;
	}
	
	/**
	 * Sets the new audio codec
	 */
	this.setAudioCodec = function (codec) {
		// Check if the codec is supported by ffmpeg version
		if (this.info_configuration.encode.indexOf(codec) != -1) {
			// Check if codec is equal 'MP3' and check if the version of ffmpeg support the libmp3lame function
			if (codec == 'mp3' && this.info_configuration.modules.indexOf('libmp3lame') != -1)
				codec = 'libmp3lame';
			
			if (options.audio == undefined)
				options.audio = new Object();
			// Set the new option
			options.audio.codec = codec;
			return this;
		} else 
			throw errors.renderError('codec_not_supported', codec);
	}
	
	/**
	 * Sets the audio sample frequency for audio outputs
	 */
	this.setAudioFrequency = function (frequency) {
		if (options.audio == undefined)
			options.audio = new Object();
		// Set the new option
		options.audio.frequency = frequency;
		return this;
	}
	
	/**
	 * Sets the number of audio channels
	 */
	this.setAudioChannels = function (channel) {
		// Check if the channel value is valid
		if (presets.audio_channel.stereo == channel || presets.audio_channel.mono == channel) {
			if (options.audio == undefined)
				options.audio = new Object();
			// Set the new option
			options.audio.channel = channel;
			return this;			
		} else 
			throw errors.renderError('audio_channel_is_invalid', channel);
	}
	
	/**
	 * Sets the audio bitrate
	 */
	this.setAudioBitRate = function (bitrate) {
		if (options.audio == undefined)
			options.audio = new Object();
		// Set the new option
		options.audio.bitrate = bitrate;
		return this;
	}
	
	/**
	 * Sets the audio quality
	 */
	this.setAudioQuality = function (quality) {
		if (options.audio == undefined)
			options.audio = new Object();
		// Set the new option
		options.audio.quality = quality;
		return this;
	}
	
	/**
	 * Sets the watermark
	 */
	this.setWatermark = function (watermarkPath, settings) {
		// Base settings
		var baseSettings = {
			position		: "SW"		// Position: NE NC NW SE SC SW C CE CW
		  , margin_nord		: null		// Margin nord
		  , margin_sud		: null		// Margin sud
		  , margin_east		: null		// Margin east
		  , margin_west		: null		// Margin west
		};
		
		// Check if watermark exists
		if (!fs.existsSync(watermarkPath))
			throw errors.renderError('invalid_watermark', watermarkPath);
		
		// Check if the settings are specified
		if (settings != null)
			utils.mergeObject(baseSettings, settings);
		
		// Check if position is valid
		if (baseSettings.position == null || utils.in_array(baseSettings.position, ['NE','NC','NW','SE','SC','SW','C','CE','CW']) === false)
			throw errors.renderError('invalid_watermark_position', baseSettings.position);
		
		// Check if margins are valid
		
		if (baseSettings.margin_nord == null || isNaN(baseSettings.margin_nord))
			baseSettings.margin_nord = 0;
		if (baseSettings.margin_sud == null || isNaN(baseSettings.margin_sud))
			baseSettings.margin_sud = 0;
		if (baseSettings.margin_east == null || isNaN(baseSettings.margin_east))
			baseSettings.margin_east = 0;
		if (baseSettings.margin_west == null || isNaN(baseSettings.margin_west))
			baseSettings.margin_west = 0;
		
		var overlay = '';
		
		var getSing = function (val, inverse) {
			return (val > 0 ? (inverse ? '-' : '+') : (inverse ? '+' : '-')).toString() + Math.abs(val).toString();
		}
		
		var getHorizontalMargins = function (east, west) {
			return getSing(east, false).toString() + getSing(west, true).toString();
		}
		
		var getVerticalMargins = function (nord, sud) {
			return getSing(nord, false).toString() + getSing(sud, true).toString();
		}
		
		// Calculate formula		
		switch (baseSettings.position) {
			case 'NE':
				overlay = '0' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':0' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
			case 'NC':
				overlay = 'main_w/2-overlay_w/2' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':0' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
			case 'NW':
				overlay = 'main_w-overlay_w' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':0' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
			case 'SE':
				overlay = '0' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':main_h-overlay_h' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
			case 'SC':
				overlay = 'main_w/2-overlay_w/2' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':main_h-overlay_h' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
			case 'SW':
				overlay = 'main_w-overlay_w' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':main_h-overlay_h' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
			case 'CE':
				overlay = '0' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':main_h/2-overlay_h/2' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
			case 'C':
				overlay = 'main_w/2-overlay_w/2' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':main_h/2-overlay_h/2' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
			case 'CW':
				overlay = 'main_w-overlay_w' + getHorizontalMargins(baseSettings.margin_east, baseSettings.margin_west) + ':main_h/2-overlay_h/2' + getVerticalMargins(baseSettings.margin_nord, baseSettings.margin_sud);
				break;
		}
		
		// Check if the call comes from internal function
		if (arguments[2] == undefined || arguments[2] == null) {
			if (options.video == undefined)
				options.video = new Object();
			// Set the new option
			options.video.watermark = { path : watermarkPath, overlay : overlay };
			return this;
		} else if (arguments[2] != undefined && arguments[2] === true) {
			this.addInput(watermarkPath);
			this.addFilterComplex('overlay=' + overlay);
		}
	}
	
	/**
	 * Save all set commands
	 */
	this.save = function (destionationFileName, callback) {
		// Check if the 'video' is present in the options
		if (options.hasOwnProperty('video')) {
			// Check if video is disabled
			if (options.video.hasOwnProperty('disabled')) {
				this.addCommand('-vn');				
			} else {
				// Check all video property
				if (options.video.hasOwnProperty('format'))
					this.addCommand('-f', options.video.format);
				if (options.video.hasOwnProperty('codec'))
					this.addCommand('-vcodec', options.video.codec);
				if (options.video.hasOwnProperty('bitrate'))
					this.addCommand('-b', parseInt(options.video.bitrate, 10) + 'kb');
				if (options.video.hasOwnProperty('framerate'))
					this.addCommand('-r', parseInt(options.video.framerate, 10));
				if (options.video.hasOwnProperty('startTime'))
					this.addCommand('-ss', parseInt(options.video.startTime, 10));
				if (options.video.hasOwnProperty('duration'))
					this.addCommand('-t', parseInt(options.video.duration, 10));
				
				if (options.video.hasOwnProperty('watermark')) {
					this.addInput(options.video.watermark.path);
					this.addFilterComplex('overlay=' + options.video.watermark.overlay);
				}
				
				// Check if the video should be scaled
				if (options.video.hasOwnProperty('size')) {
					var newDimension = _calculateNewDimension.call(this);
					
					if (newDimension.aspect != null) {
						this.addFilterComplex('scale=iw*sar:ih, pad=max(iw\\,ih*(' + newDimension.aspect.x + '/' + newDimension.aspect.y + ')):ow/(' + newDimension.aspect.x + '/' + newDimension.aspect.y + '):(ow-iw)/2:(oh-ih)/2' + (options.video.paddingColor != null ? ':' + options.video.paddingColor : ''));
						this.addCommand('-aspect', newDimension.aspect.string);
					}
					
					this.addCommand('-s', newDimension.width + 'x' + newDimension.height);
				}
			}
		}
		// Check if the 'audio' is present in the options
		if (options.hasOwnProperty('audio')) {
			// Check if audio is disabled
			if (options.audio.hasOwnProperty('disabled')) {
				this.addCommand('-an');				
			} else {
				// Check all audio property
				if (options.audio.hasOwnProperty('codec'))
					this.addCommand('-acodec', options.audio.codec);
				if (options.audio.hasOwnProperty('frequency'))
					this.addCommand('-ar', parseInt(options.audio.frequency));
				if (options.audio.hasOwnProperty('channel'))
					this.addCommand('-ac', options.audio.channel);
				if (options.audio.hasOwnProperty('quality'))
					this.addCommand('-aq', options.audio.quality);
				if (options.audio.hasOwnProperty('bitrate'))
					this.addCommand('-ab', parseInt(options.audio.bitrate, 10) + 'k');
			}
		}
		
		setOutput(destionationFileName);
		
		return execCommand.call(this, callback);
	}
	
	/*********************/
	/* INTERNAL FUNCTION */
	/*********************/
	
	/**
	 * Reset the list of commands
	 */
	var resetCommands = function (self) {
		commands		= new Array()
		inputs			= [self.file_path];
		filtersComlpex	= new Array();
		output			= null;
		options			= new Object();
	}

	/**
	 * Calculate width, height and aspect ratio by the new dimension data
	 */
	var _calculateNewDimension = function () {
		// Check if keepPixelAspectRatio is undefined
		var keepPixelAspectRatio = typeof options.video.keepPixelAspectRatio != 'boolean' ? false : options.video.keepPixelAspectRatio;
		// Check if keepAspectRatio is undefined
		var keepAspectRatio = typeof options.video.keepAspectRatio != 'boolean' ? false : options.video.keepAspectRatio;
		
		// Resolution to be taken as a reference
		var referrerResolution = this.metadata.video.resolution;
		// Check if is need keep pixel aspect ratio
		if (keepPixelAspectRatio) {
			// Check if exists resolution for pixel aspect ratio
			if (utils.isEmptyObj(this.metadata.video.resolutionSquare))
				throw errors.renderError('resolution_square_not_defined');
			
			// Apply the resolutionSquare
			referrerResolution = this.metadata.video.resolutionSquare;
		}
		
		// Final data
		var width	= null
		  , height	= null
		  , aspect	= null;

		// Regex to check which type of dimension was specified
		var fixedWidth		= /([0-9]+)x\?/.exec(options.video.size)
		  , fixedHeight		= /\?x([0-9]+)/.exec(options.video.size)
		  , percentage		= /([0-9]{1,2})%/.exec(options.video.size)
		  , classicSize		= /([0-9]+)x([0-9]+)/.exec(options.video.size);
		  
		if (fixedWidth) {
			// Set the width dimension
			width = parseInt(fixedWidth[1], 10);			
			// Check if the video has the aspect ratio setted
			if (!utils.isEmptyObj(this.metadata.video.aspect)) {
				height = Math.round((width / this.metadata.video.aspect.x) * this.metadata.video.aspect.y);
			} else {
				// Calculte the new height
				height = Math.round(referrerResolution.h / (referrerResolution.w / parseInt(fixedWidth[1], 10)));
			}
		} else if (fixedHeight) {
			// Set the width dimension
			height = parseInt(fixedHeight[1], 10);			
			// Check if the video has the aspect ratio setted
			if (!utils.isEmptyObj(this.metadata.video.aspect)) {
				width = Math.round((height / this.metadata.video.aspect.y) * this.metadata.video.aspect.x);
			} else {
				// Calculte the new width
				width = Math.round(referrerResolution.w / (referrerResolution.h / parseInt(fixedHeight[1], 10)));
			}			
		} else if (percentage) {
			// Calculte the ratio from percentage
			var ratio = parseInt(percentage[1], 10) / 100;
			// Calculate the new dimensions
			width = Math.round(referrerResolution.w * ratio);
			height = Math.round(referrerResolution.h * ratio);
		} else if (classicSize) {
			width = parseInt(classicSize[1], 10);
			height = parseInt(classicSize[2], 10);
		} else 
			throw errors.renderError('size_format', options.video.size);
		
		// If the width or height are not multiples of 2 will be decremented by one unit
		if (width % 2 != 0) width -= 1;
		if (height % 2 != 0) height -= 1;
		
		if (keepAspectRatio) {
			// Calculate the new aspect ratio
			var gcdValue	= utils.gcd(width, height);
			
			aspect = new Object();
			aspect.x = width / gcdValue;
			aspect.y = height / gcdValue;
			aspect.string = aspect.x + ':' + aspect.y;
		}
		
		return { width : width, height : height, aspect : aspect };
	}
	
	/**
	 * Executing the commands list
	 */
	var execCommand = function (callback, folder) {
		// Checking if folder is defined
		var onlyDestinationFile = folder != undefined ? false : true;
		// Building the value for return value. Check if the callback is not a function. In this case will created a new instance of the deferred class
		var deferred = typeof callback != 'function' ? when.defer() : { promise : null };
		// Create a copy of the commands list
		var finalCommands = ['ffmpeg -i']
			.concat(inputs.join(' -i '))
			.concat(commands.join(' '))
			.concat(filtersComlpex.length > 0 ? ['-filter_complex "'].concat(filtersComlpex.join(', ')).join('') + '"' : [])
			.concat([output]);
		// Reset commands
		resetCommands(this);
		// Execute the commands from the list
		utils.exec(finalCommands, settings, function (error, stdout, stderr) {
			// Building the result
			var result = null;
			if (!error) {
				// Check if show only destination filename or the complete file list
				if (onlyDestinationFile) {
					result = finalCommands[finalCommands.length-1];
				} else {
					// Clean possible "/" at the end of the string
					if (folder.charAt(folder.length-1) == "/")
						folder = folder.substr(0, folder.length-1);
					// Read file list inside the folder
					result = fs.readdirSync(folder);
					// Scan all file and prepend the folder path
					for (var i in result)
						result[i] = [folder, result[i]].join('/')
				}
			}
			// Check if the callback is a function
			if (typeof callback == 'function') {
				// Call the callback to return the info
				callback(error, result);
			} else {
				if (error) {
					// Negative response
					deferred.reject(error);
				} else {
					// Positive response
					deferred.resolve(result);
				}
			}
		});
		// Return a possible promise instance
		return deferred.promise;
	}
	
	/*******************/
	/* PRESET FUNCTION */
	/*******************/
	
	/**
	 * Extracting sound from a video, and save it as Mp3
	 */
	this.fnExtractSoundToMP3 = function (destionationFileName, callback) {
		// Check if file already exists. In this case will remove it
		if (fs.existsSync(destionationFileName)) 
			fs.unlinkSync(destionationFileName);

		// Building the final path
		var destinationDirName		= path.dirname(destionationFileName)
		  , destinationFileNameWE	= path.basename(destionationFileName, path.extname(destionationFileName)) + '.mp3'
		  , finalPath				= path.join(destinationDirName, destinationFileNameWE);
		
		resetCommands(this);
		
		// Adding commands to the list
		this.addCommand('-vn');
		this.addCommand('-ar', 44100);
		this.addCommand('-ac', 2);
		this.addCommand('-ab', 192);
		this.addCommand('-f', 'mp3');
		
		// Add destination file path to the command list
		setOutput(finalPath);
		
		// Executing the commands list
		return execCommand.call(this, callback);
	}
	
	/**
	 * Extract frame from video file
	 */
	this.fnExtractFrameToJPG = function (/* destinationFolder, settings, callback */) {
		
		var destinationFolder	= null
		  , newSettings			= null
		  , callback			= null;
		  
		var settings = {
			start_time				: null		// Start time to recording
		  , duration_time			: null		// Duration of recording
		  , frame_rate				: null		// Number of the frames to capture in one second
		  , size					: null		// Dimension each frame
		  , number					: null		// Total frame to capture
		  , every_n_frames			: null		// Frame to capture every N frames
		  , every_n_seconds			: null		// Frame to capture every N seconds
		  , every_n_percentage		: null		// Frame to capture every N percentage range
		  , keep_pixel_aspect_ratio	: true		// Mantain the original pixel video aspect ratio
		  , keep_aspect_ratio		: true		// Mantain the original aspect ratio
		  , padding_color			: 'black'	// Padding color
		  , file_name				: null		// File name
		};
		  
		// Scan all arguments
		for (var i in arguments) {
			// Check the type of the argument
			switch (typeof arguments[i]) {
				case 'string':
					destinationFolder = arguments[i];
					break;
				case 'object':
					newSettings = arguments[i];
					break;
				case 'function':
					callback = arguments[i];
					break;
			}
		}
		
		// Check if the settings are specified
		if (newSettings !== null)
			utils.mergeObject(settings, newSettings);

		// Check if 'start_time' is in the format hours:minutes:seconds
		if (settings.start_time != null) {
			if (/([0-9]+):([0-9]{2}):([0-9]{2})/.exec(settings.start_time))
				settings.start_time = utils.durationToSeconds(settings.start_time);
			else if (!isNaN(settings.start_time))
				settings.start_time = parseInt(settings.start_time, 10);
			else
				settings.start_time = null;
		}

		// Check if 'duration_time' is in the format hours:minutes:seconds
		if (settings.duration_time != null) {
			if (/([0-9]+):([0-9]{2}):([0-9]{2})/.exec(settings.duration_time))
				settings.duration_time = utils.durationToSeconds(settings.duration_time);
			else if (!isNaN(settings.duration_time))
				settings.duration_time = parseInt(settings.duration_time, 10);
			else
				settings.duration_time = null;
		}

		// Check if the value of the framerate is number type
		if (settings.frame_rate != null && isNaN(settings.frame_rate))
			settings.frame_rate = null;

		// If the size is not settings then the size of the screenshots is equal to video size
		if (settings.size == null)
			settings.size = this.metadata.video.resolution.w + 'x' + this.metadata.video.resolution.h;

		// Check if the value of the 'number frame to capture' is number type
		if (settings.number != null && isNaN(settings.number))
			settings.number = null;

		var every_n_check = 0;

		// Check if the value of the 'every_n_frames' is number type
		if (settings.every_n_frames != null && isNaN(settings.every_n_frames)) {
			settings.every_n_frames = null;
			every_n_check++;
		}

		// Check if the value of the 'every_n_seconds' is number type
		if (settings.every_n_seconds != null && isNaN(settings.every_n_seconds)) {
			settings.every_n_seconds = null;
			every_n_check++;
		}

		// Check if the value of the 'every_n_percentage' is number type
		if (settings.every_n_percentage != null && (isNaN(settings.every_n_percentage) || settings.every_n_percentage > 100)) {
			settings.every_n_percentage = null;
			every_n_check++;
		}
		
		if (every_n_check >= 2) {
			if (callback) {
				callback(errors.renderError('extract_frame_invalid_everyN_options'));
			} else {
				throw errors.renderError('extract_frame_invalid_everyN_options');
			}
		}			
		
		// If filename is null then his value is equal to original filename
		if (settings.file_name == null) {
			settings.file_name = path.basename(this.file_path, path.extname(this.file_path));
		} else {
			// Retrieve all possible replacements
			var replacements = settings.file_name.match(/(\%[a-zA-Z]{1})/g);
			// Check if exists replacements. The scan all replacements and build the final filename
			if (replacements) {
				for (var i in replacements) {
					switch (replacements[i]) {
						case '%t':
							settings.file_name = settings.file_name.replace('%t', new Date().getTime());
							break;
						case '%s':
							settings.file_name = settings.file_name.replace('%s', settings.size);
							break;
						case '%x':
							settings.file_name = settings.file_name.replace('%x', settings.size.split(':')[0]);
							break;
						case '%y':
							settings.file_name = settings.file_name.replace('%y', settings.size.split(':')[1]);
							break;
						default:
							settings.file_name = settings.file_name.replace(replacements[i], '');
							break;
					}
				}
			}
		}
		// At the filename will added the number of the frame
		settings.file_name = path.basename(settings.file_name, path.extname(settings.file_name)) + '_%d.jpg';
		
		// Create the directory to save the extracted frames
		utils.mkdir(destinationFolder, 0777);
		
		resetCommands(this);
		
		// Adding commands to the list
		if (settings.startTime)
			this.addCommand('-ss', settings.startTime);
		if (settings.duration_time)
			this.addCommand('-t', settings.duration_time);
		if (settings.frame_rate)
			this.addCommand('-r', settings.frame_rate);

		// Setting the size and padding settings
		this.setVideoSize(settings.size, settings.keep_pixel_aspect_ratio, settings.keep_aspect_ratio, settings.padding_color);
		// Get the dimensions
		var newDimension = _calculateNewDimension.call(this);
		// Apply the size and padding commands
		this.addCommand('-s', newDimension.width + 'x' + newDimension.height);
		// CHeck if isset aspect ratio options
		if (newDimension.aspect != null) {
			this.addFilterComplex('scale=iw*sar:ih, pad=max(iw\\,ih*(' + newDimension.aspect.x + '/' + newDimension.aspect.y + ')):ow/(' + newDimension.aspect.x + '/' + newDimension.aspect.y + '):(ow-iw)/2:(oh-ih)/2' + (settings.padding_color != null ? ':' + settings.padding_color : ''));
			this.addCommand('-aspect', newDimension.aspect.string);
		}

		if (settings.number)
			this.addCommand('-vframes', settings.number);
		if (settings.every_n_frames) {
			this.addCommand('-vsync', 0);					
			this.addFilterComplex('select=not(mod(n\\,' + settings.every_n_frames + '))');
		}
		if (settings.every_n_seconds) {
			this.addCommand('-vsync', 0);
			this.addFilterComplex('select=not(mod(t\\,' + settings.every_n_seconds + '))');
		}
		if (settings.every_n_percentage) {
			this.addCommand('-vsync', 0);
			this.addFilterComplex('select=not(mod(t\\,' + parseInt((this.metadata.duration.seconds / 100) * settings.every_n_percentage) + '))');
		}
		
		// Add destination file path to the command list
		setOutput([destinationFolder,settings.file_name].join('/'));

		// Executing the commands list
		return execCommand.call(this, callback, destinationFolder);
	}

	/**
	 * Add a watermark to the video and save it
	 */
	this.fnAddWatermark = function (watermarkPath /* newFilepath , settings, callback */) {

		var newFilepath		= null
		  , newSettings		= null
		  , callback		= null;
		  
		// Scan all arguments
		for (var i = 1; i < arguments.length; i++) {
			// Check the type of the argument
			switch (typeof arguments[i]) {
				case 'string':
					newFilepath = arguments[i];
					break;
				case 'object':
					newSettings = arguments[i];
					break;
				case 'function':
					callback = arguments[i];
					break;
			}
		}
		
		resetCommands(this);

		// Call the function to add the watermark options
		this.setWatermark(watermarkPath, newSettings, true);
		
		if (newFilepath == null)
			newFilepath = path.dirname(this.file_path) + '/' + 
						  path.basename(this.file_path, path.extname(this.file_path)) + '_watermark_' + 
						  path.basename(watermarkPath, path.extname(watermarkPath)) + 
						  path.extname(this.file_path);
		
		// Add destination file path to the command list
		setOutput(newFilepath);

		// Executing the commands list
		return execCommand.call(this, callback);
	}
	
	/**
	 * Constructor
	 */
	var __constructor = function (self) {
		resetCommands(self);
	}(this);
}
},{"./errors":4,"./presets":6,"./utils":7,"fs":27,"path":28,"when":26}],9:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function (require) {

	var makePromise = require('./makePromise');
	var Scheduler = require('./Scheduler');
	var async = require('./env').asap;

	return makePromise({
		scheduler: new Scheduler(async)
	});

});
})(typeof define === 'function' && define.amd ? define : function (factory) { module.exports = factory(require); });

},{"./Scheduler":10,"./env":22,"./makePromise":24}],10:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	// Credit to Twisol (https://github.com/Twisol) for suggesting
	// this type of extensible queue + trampoline approach for next-tick conflation.

	/**
	 * Async task scheduler
	 * @param {function} async function to schedule a single async function
	 * @constructor
	 */
	function Scheduler(async) {
		this._async = async;
		this._running = false;

		this._queue = this;
		this._queueLen = 0;
		this._afterQueue = {};
		this._afterQueueLen = 0;

		var self = this;
		this.drain = function() {
			self._drain();
		};
	}

	/**
	 * Enqueue a task
	 * @param {{ run:function }} task
	 */
	Scheduler.prototype.enqueue = function(task) {
		this._queue[this._queueLen++] = task;
		this.run();
	};

	/**
	 * Enqueue a task to run after the main task queue
	 * @param {{ run:function }} task
	 */
	Scheduler.prototype.afterQueue = function(task) {
		this._afterQueue[this._afterQueueLen++] = task;
		this.run();
	};

	Scheduler.prototype.run = function() {
		if (!this._running) {
			this._running = true;
			this._async(this.drain);
		}
	};

	/**
	 * Drain the handler queue entirely, and then the after queue
	 */
	Scheduler.prototype._drain = function() {
		var i = 0;
		for (; i < this._queueLen; ++i) {
			this._queue[i].run();
			this._queue[i] = void 0;
		}

		this._queueLen = 0;
		this._running = false;

		for (i = 0; i < this._afterQueueLen; ++i) {
			this._afterQueue[i].run();
			this._afterQueue[i] = void 0;
		}

		this._afterQueueLen = 0;
	};

	return Scheduler;

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));

},{}],11:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	/**
	 * Custom error type for promises rejected by promise.timeout
	 * @param {string} message
	 * @constructor
	 */
	function TimeoutError (message) {
		Error.call(this);
		this.message = message;
		this.name = TimeoutError.name;
		if (typeof Error.captureStackTrace === 'function') {
			Error.captureStackTrace(this, TimeoutError);
		}
	}

	TimeoutError.prototype = Object.create(Error.prototype);
	TimeoutError.prototype.constructor = TimeoutError;

	return TimeoutError;
});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));
},{}],12:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	makeApply.tryCatchResolve = tryCatchResolve;

	return makeApply;

	function makeApply(Promise, call) {
		if(arguments.length < 2) {
			call = tryCatchResolve;
		}

		return apply;

		function apply(f, thisArg, args) {
			var p = Promise._defer();
			var l = args.length;
			var params = new Array(l);
			callAndResolve({ f:f, thisArg:thisArg, args:args, params:params, i:l-1, call:call }, p._handler);

			return p;
		}

		function callAndResolve(c, h) {
			if(c.i < 0) {
				return call(c.f, c.thisArg, c.params, h);
			}

			var handler = Promise._handler(c.args[c.i]);
			handler.fold(callAndResolveNext, c, void 0, h);
		}

		function callAndResolveNext(c, x, h) {
			c.params[c.i] = x;
			c.i -= 1;
			callAndResolve(c, h);
		}
	}

	function tryCatchResolve(f, thisArg, args, resolver) {
		try {
			resolver.resolve(f.apply(thisArg, args));
		} catch(e) {
			resolver.reject(e);
		}
	}

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));



},{}],13:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function(require) {

	var state = require('../state');
	var applier = require('../apply');

	return function array(Promise) {

		var applyFold = applier(Promise);
		var toPromise = Promise.resolve;
		var all = Promise.all;

		var ar = Array.prototype.reduce;
		var arr = Array.prototype.reduceRight;
		var slice = Array.prototype.slice;

		// Additional array combinators

		Promise.any = any;
		Promise.some = some;
		Promise.settle = settle;

		Promise.map = map;
		Promise.filter = filter;
		Promise.reduce = reduce;
		Promise.reduceRight = reduceRight;

		/**
		 * When this promise fulfills with an array, do
		 * onFulfilled.apply(void 0, array)
		 * @param {function} onFulfilled function to apply
		 * @returns {Promise} promise for the result of applying onFulfilled
		 */
		Promise.prototype.spread = function(onFulfilled) {
			return this.then(all).then(function(array) {
				return onFulfilled.apply(this, array);
			});
		};

		return Promise;

		/**
		 * One-winner competitive race.
		 * Return a promise that will fulfill when one of the promises
		 * in the input array fulfills, or will reject when all promises
		 * have rejected.
		 * @param {array} promises
		 * @returns {Promise} promise for the first fulfilled value
		 */
		function any(promises) {
			var p = Promise._defer();
			var resolver = p._handler;
			var l = promises.length>>>0;

			var pending = l;
			var errors = [];

			for (var h, x, i = 0; i < l; ++i) {
				x = promises[i];
				if(x === void 0 && !(i in promises)) {
					--pending;
					continue;
				}

				h = Promise._handler(x);
				if(h.state() > 0) {
					resolver.become(h);
					Promise._visitRemaining(promises, i, h);
					break;
				} else {
					h.visit(resolver, handleFulfill, handleReject);
				}
			}

			if(pending === 0) {
				resolver.reject(new RangeError('any(): array must not be empty'));
			}

			return p;

			function handleFulfill(x) {
				/*jshint validthis:true*/
				errors = null;
				this.resolve(x); // this === resolver
			}

			function handleReject(e) {
				/*jshint validthis:true*/
				if(this.resolved) { // this === resolver
					return;
				}

				errors.push(e);
				if(--pending === 0) {
					this.reject(errors);
				}
			}
		}

		/**
		 * N-winner competitive race
		 * Return a promise that will fulfill when n input promises have
		 * fulfilled, or will reject when it becomes impossible for n
		 * input promises to fulfill (ie when promises.length - n + 1
		 * have rejected)
		 * @param {array} promises
		 * @param {number} n
		 * @returns {Promise} promise for the earliest n fulfillment values
		 *
		 * @deprecated
		 */
		function some(promises, n) {
			/*jshint maxcomplexity:7*/
			var p = Promise._defer();
			var resolver = p._handler;

			var results = [];
			var errors = [];

			var l = promises.length>>>0;
			var nFulfill = 0;
			var nReject;
			var x, i; // reused in both for() loops

			// First pass: count actual array items
			for(i=0; i<l; ++i) {
				x = promises[i];
				if(x === void 0 && !(i in promises)) {
					continue;
				}
				++nFulfill;
			}

			// Compute actual goals
			n = Math.max(n, 0);
			nReject = (nFulfill - n + 1);
			nFulfill = Math.min(n, nFulfill);

			if(n > nFulfill) {
				resolver.reject(new RangeError('some(): array must contain at least '
				+ n + ' item(s), but had ' + nFulfill));
			} else if(nFulfill === 0) {
				resolver.resolve(results);
			}

			// Second pass: observe each array item, make progress toward goals
			for(i=0; i<l; ++i) {
				x = promises[i];
				if(x === void 0 && !(i in promises)) {
					continue;
				}

				Promise._handler(x).visit(resolver, fulfill, reject, resolver.notify);
			}

			return p;

			function fulfill(x) {
				/*jshint validthis:true*/
				if(this.resolved) { // this === resolver
					return;
				}

				results.push(x);
				if(--nFulfill === 0) {
					errors = null;
					this.resolve(results);
				}
			}

			function reject(e) {
				/*jshint validthis:true*/
				if(this.resolved) { // this === resolver
					return;
				}

				errors.push(e);
				if(--nReject === 0) {
					results = null;
					this.reject(errors);
				}
			}
		}

		/**
		 * Apply f to the value of each promise in a list of promises
		 * and return a new list containing the results.
		 * @param {array} promises
		 * @param {function(x:*, index:Number):*} f mapping function
		 * @returns {Promise}
		 */
		function map(promises, f) {
			return Promise._traverse(f, promises);
		}

		/**
		 * Filter the provided array of promises using the provided predicate.  Input may
		 * contain promises and values
		 * @param {Array} promises array of promises and values
		 * @param {function(x:*, index:Number):boolean} predicate filtering predicate.
		 *  Must return truthy (or promise for truthy) for items to retain.
		 * @returns {Promise} promise that will fulfill with an array containing all items
		 *  for which predicate returned truthy.
		 */
		function filter(promises, predicate) {
			var a = slice.call(promises);
			return Promise._traverse(predicate, a).then(function(keep) {
				return filterSync(a, keep);
			});
		}

		function filterSync(promises, keep) {
			// Safe because we know all promises have fulfilled if we've made it this far
			var l = keep.length;
			var filtered = new Array(l);
			for(var i=0, j=0; i<l; ++i) {
				if(keep[i]) {
					filtered[j++] = Promise._handler(promises[i]).value;
				}
			}
			filtered.length = j;
			return filtered;

		}

		/**
		 * Return a promise that will always fulfill with an array containing
		 * the outcome states of all input promises.  The returned promise
		 * will never reject.
		 * @param {Array} promises
		 * @returns {Promise} promise for array of settled state descriptors
		 */
		function settle(promises) {
			return all(promises.map(settleOne));
		}

		function settleOne(p) {
			// Optimize the case where we get an already-resolved when.js promise
			//  by extracting its state:
			var handler;
			if (p instanceof Promise) {
				// This is our own Promise type and we can reach its handler internals:
				handler = p._handler.join();
			}
			if((handler && handler.state() === 0) || !handler) {
				// Either still pending, or not a Promise at all:
				return toPromise(p).then(state.fulfilled, state.rejected);
			}

			// The promise is our own, but it is already resolved. Take a shortcut.
			// Since we're not actually handling the resolution, we need to disable
			// rejection reporting.
			handler._unreport();
			return state.inspect(handler);
		}

		/**
		 * Traditional reduce function, similar to `Array.prototype.reduce()`, but
		 * input may contain promises and/or values, and reduceFunc
		 * may return either a value or a promise, *and* initialValue may
		 * be a promise for the starting value.
		 * @param {Array|Promise} promises array or promise for an array of anything,
		 *      may contain a mix of promises and values.
		 * @param {function(accumulated:*, x:*, index:Number):*} f reduce function
		 * @returns {Promise} that will resolve to the final reduced value
		 */
		function reduce(promises, f /*, initialValue */) {
			return arguments.length > 2 ? ar.call(promises, liftCombine(f), arguments[2])
					: ar.call(promises, liftCombine(f));
		}

		/**
		 * Traditional reduce function, similar to `Array.prototype.reduceRight()`, but
		 * input may contain promises and/or values, and reduceFunc
		 * may return either a value or a promise, *and* initialValue may
		 * be a promise for the starting value.
		 * @param {Array|Promise} promises array or promise for an array of anything,
		 *      may contain a mix of promises and values.
		 * @param {function(accumulated:*, x:*, index:Number):*} f reduce function
		 * @returns {Promise} that will resolve to the final reduced value
		 */
		function reduceRight(promises, f /*, initialValue */) {
			return arguments.length > 2 ? arr.call(promises, liftCombine(f), arguments[2])
					: arr.call(promises, liftCombine(f));
		}

		function liftCombine(f) {
			return function(z, x, i) {
				return applyFold(f, void 0, [z,x,i]);
			};
		}
	};

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(require); }));

},{"../apply":12,"../state":25}],14:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	return function flow(Promise) {

		var resolve = Promise.resolve;
		var reject = Promise.reject;
		var origCatch = Promise.prototype['catch'];

		/**
		 * Handle the ultimate fulfillment value or rejection reason, and assume
		 * responsibility for all errors.  If an error propagates out of result
		 * or handleFatalError, it will be rethrown to the host, resulting in a
		 * loud stack track on most platforms and a crash on some.
		 * @param {function?} onResult
		 * @param {function?} onError
		 * @returns {undefined}
		 */
		Promise.prototype.done = function(onResult, onError) {
			this._handler.visit(this._handler.receiver, onResult, onError);
		};

		/**
		 * Add Error-type and predicate matching to catch.  Examples:
		 * promise.catch(TypeError, handleTypeError)
		 *   .catch(predicate, handleMatchedErrors)
		 *   .catch(handleRemainingErrors)
		 * @param onRejected
		 * @returns {*}
		 */
		Promise.prototype['catch'] = Promise.prototype.otherwise = function(onRejected) {
			if (arguments.length < 2) {
				return origCatch.call(this, onRejected);
			}

			if(typeof onRejected !== 'function') {
				return this.ensure(rejectInvalidPredicate);
			}

			return origCatch.call(this, createCatchFilter(arguments[1], onRejected));
		};

		/**
		 * Wraps the provided catch handler, so that it will only be called
		 * if the predicate evaluates truthy
		 * @param {?function} handler
		 * @param {function} predicate
		 * @returns {function} conditional catch handler
		 */
		function createCatchFilter(handler, predicate) {
			return function(e) {
				return evaluatePredicate(e, predicate)
					? handler.call(this, e)
					: reject(e);
			};
		}

		/**
		 * Ensures that onFulfilledOrRejected will be called regardless of whether
		 * this promise is fulfilled or rejected.  onFulfilledOrRejected WILL NOT
		 * receive the promises' value or reason.  Any returned value will be disregarded.
		 * onFulfilledOrRejected may throw or return a rejected promise to signal
		 * an additional error.
		 * @param {function} handler handler to be called regardless of
		 *  fulfillment or rejection
		 * @returns {Promise}
		 */
		Promise.prototype['finally'] = Promise.prototype.ensure = function(handler) {
			if(typeof handler !== 'function') {
				return this;
			}

			return this.then(function(x) {
				return runSideEffect(handler, this, identity, x);
			}, function(e) {
				return runSideEffect(handler, this, reject, e);
			});
		};

		function runSideEffect (handler, thisArg, propagate, value) {
			var result = handler.call(thisArg);
			return maybeThenable(result)
				? propagateValue(result, propagate, value)
				: propagate(value);
		}

		function propagateValue (result, propagate, x) {
			return resolve(result).then(function () {
				return propagate(x);
			});
		}

		/**
		 * Recover from a failure by returning a defaultValue.  If defaultValue
		 * is a promise, it's fulfillment value will be used.  If defaultValue is
		 * a promise that rejects, the returned promise will reject with the
		 * same reason.
		 * @param {*} defaultValue
		 * @returns {Promise} new promise
		 */
		Promise.prototype['else'] = Promise.prototype.orElse = function(defaultValue) {
			return this.then(void 0, function() {
				return defaultValue;
			});
		};

		/**
		 * Shortcut for .then(function() { return value; })
		 * @param  {*} value
		 * @return {Promise} a promise that:
		 *  - is fulfilled if value is not a promise, or
		 *  - if value is a promise, will fulfill with its value, or reject
		 *    with its reason.
		 */
		Promise.prototype['yield'] = function(value) {
			return this.then(function() {
				return value;
			});
		};

		/**
		 * Runs a side effect when this promise fulfills, without changing the
		 * fulfillment value.
		 * @param {function} onFulfilledSideEffect
		 * @returns {Promise}
		 */
		Promise.prototype.tap = function(onFulfilledSideEffect) {
			return this.then(onFulfilledSideEffect)['yield'](this);
		};

		return Promise;
	};

	function rejectInvalidPredicate() {
		throw new TypeError('catch predicate must be a function');
	}

	function evaluatePredicate(e, predicate) {
		return isError(predicate) ? e instanceof predicate : predicate(e);
	}

	function isError(predicate) {
		return predicate === Error
			|| (predicate != null && predicate.prototype instanceof Error);
	}

	function maybeThenable(x) {
		return (typeof x === 'object' || typeof x === 'function') && x !== null;
	}

	function identity(x) {
		return x;
	}

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));

},{}],15:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */
/** @author Jeff Escalante */

(function(define) { 'use strict';
define(function() {

	return function fold(Promise) {

		Promise.prototype.fold = function(f, z) {
			var promise = this._beget();

			this._handler.fold(function(z, x, to) {
				Promise._handler(z).fold(function(x, z, to) {
					to.resolve(f.call(this, z, x));
				}, x, this, to);
			}, z, promise._handler.receiver, promise._handler);

			return promise;
		};

		return Promise;
	};

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));

},{}],16:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function(require) {

	var inspect = require('../state').inspect;

	return function inspection(Promise) {

		Promise.prototype.inspect = function() {
			return inspect(Promise._handler(this));
		};

		return Promise;
	};

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(require); }));

},{"../state":25}],17:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	return function generate(Promise) {

		var resolve = Promise.resolve;

		Promise.iterate = iterate;
		Promise.unfold = unfold;

		return Promise;

		/**
		 * @deprecated Use github.com/cujojs/most streams and most.iterate
		 * Generate a (potentially infinite) stream of promised values:
		 * x, f(x), f(f(x)), etc. until condition(x) returns true
		 * @param {function} f function to generate a new x from the previous x
		 * @param {function} condition function that, given the current x, returns
		 *  truthy when the iterate should stop
		 * @param {function} handler function to handle the value produced by f
		 * @param {*|Promise} x starting value, may be a promise
		 * @return {Promise} the result of the last call to f before
		 *  condition returns true
		 */
		function iterate(f, condition, handler, x) {
			return unfold(function(x) {
				return [x, f(x)];
			}, condition, handler, x);
		}

		/**
		 * @deprecated Use github.com/cujojs/most streams and most.unfold
		 * Generate a (potentially infinite) stream of promised values
		 * by applying handler(generator(seed)) iteratively until
		 * condition(seed) returns true.
		 * @param {function} unspool function that generates a [value, newSeed]
		 *  given a seed.
		 * @param {function} condition function that, given the current seed, returns
		 *  truthy when the unfold should stop
		 * @param {function} handler function to handle the value produced by unspool
		 * @param x {*|Promise} starting value, may be a promise
		 * @return {Promise} the result of the last value produced by unspool before
		 *  condition returns true
		 */
		function unfold(unspool, condition, handler, x) {
			return resolve(x).then(function(seed) {
				return resolve(condition(seed)).then(function(done) {
					return done ? seed : resolve(unspool(seed)).spread(next);
				});
			});

			function next(item, newSeed) {
				return resolve(handler(item)).then(function() {
					return unfold(unspool, condition, handler, newSeed);
				});
			}
		}
	};

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));

},{}],18:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	return function progress(Promise) {

		/**
		 * @deprecated
		 * Register a progress handler for this promise
		 * @param {function} onProgress
		 * @returns {Promise}
		 */
		Promise.prototype.progress = function(onProgress) {
			return this.then(void 0, void 0, onProgress);
		};

		return Promise;
	};

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));

},{}],19:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function(require) {

	var env = require('../env');
	var TimeoutError = require('../TimeoutError');

	function setTimeout(f, ms, x, y) {
		return env.setTimer(function() {
			f(x, y, ms);
		}, ms);
	}

	return function timed(Promise) {
		/**
		 * Return a new promise whose fulfillment value is revealed only
		 * after ms milliseconds
		 * @param {number} ms milliseconds
		 * @returns {Promise}
		 */
		Promise.prototype.delay = function(ms) {
			var p = this._beget();
			this._handler.fold(handleDelay, ms, void 0, p._handler);
			return p;
		};

		function handleDelay(ms, x, h) {
			setTimeout(resolveDelay, ms, x, h);
		}

		function resolveDelay(x, h) {
			h.resolve(x);
		}

		/**
		 * Return a new promise that rejects after ms milliseconds unless
		 * this promise fulfills earlier, in which case the returned promise
		 * fulfills with the same value.
		 * @param {number} ms milliseconds
		 * @param {Error|*=} reason optional rejection reason to use, defaults
		 *   to a TimeoutError if not provided
		 * @returns {Promise}
		 */
		Promise.prototype.timeout = function(ms, reason) {
			var p = this._beget();
			var h = p._handler;

			var t = setTimeout(onTimeout, ms, reason, p._handler);

			this._handler.visit(h,
				function onFulfill(x) {
					env.clearTimer(t);
					this.resolve(x); // this = h
				},
				function onReject(x) {
					env.clearTimer(t);
					this.reject(x); // this = h
				},
				h.notify);

			return p;
		};

		function onTimeout(reason, h, ms) {
			var e = typeof reason === 'undefined'
				? new TimeoutError('timed out after ' + ms + 'ms')
				: reason;
			h.reject(e);
		}

		return Promise;
	};

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(require); }));

},{"../TimeoutError":11,"../env":22}],20:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function(require) {

	var setTimer = require('../env').setTimer;
	var format = require('../format');

	return function unhandledRejection(Promise) {

		var logError = noop;
		var logInfo = noop;
		var localConsole;

		if(typeof console !== 'undefined') {
			// Alias console to prevent things like uglify's drop_console option from
			// removing console.log/error. Unhandled rejections fall into the same
			// category as uncaught exceptions, and build tools shouldn't silence them.
			localConsole = console;
			logError = typeof localConsole.error !== 'undefined'
				? function (e) { localConsole.error(e); }
				: function (e) { localConsole.log(e); };

			logInfo = typeof localConsole.info !== 'undefined'
				? function (e) { localConsole.info(e); }
				: function (e) { localConsole.log(e); };
		}

		Promise.onPotentiallyUnhandledRejection = function(rejection) {
			enqueue(report, rejection);
		};

		Promise.onPotentiallyUnhandledRejectionHandled = function(rejection) {
			enqueue(unreport, rejection);
		};

		Promise.onFatalRejection = function(rejection) {
			enqueue(throwit, rejection.value);
		};

		var tasks = [];
		var reported = [];
		var running = null;

		function report(r) {
			if(!r.handled) {
				reported.push(r);
				logError('Potentially unhandled rejection [' + r.id + '] ' + format.formatError(r.value));
			}
		}

		function unreport(r) {
			var i = reported.indexOf(r);
			if(i >= 0) {
				reported.splice(i, 1);
				logInfo('Handled previous rejection [' + r.id + '] ' + format.formatObject(r.value));
			}
		}

		function enqueue(f, x) {
			tasks.push(f, x);
			if(running === null) {
				running = setTimer(flush, 0);
			}
		}

		function flush() {
			running = null;
			while(tasks.length > 0) {
				tasks.shift()(tasks.shift());
			}
		}

		return Promise;
	};

	function throwit(e) {
		throw e;
	}

	function noop() {}

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(require); }));

},{"../env":22,"../format":23}],21:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	return function addWith(Promise) {
		/**
		 * Returns a promise whose handlers will be called with `this` set to
		 * the supplied receiver.  Subsequent promises derived from the
		 * returned promise will also have their handlers called with receiver
		 * as `this`. Calling `with` with undefined or no arguments will return
		 * a promise whose handlers will again be called in the usual Promises/A+
		 * way (no `this`) thus safely undoing any previous `with` in the
		 * promise chain.
		 *
		 * WARNING: Promises returned from `with`/`withThis` are NOT Promises/A+
		 * compliant, specifically violating 2.2.5 (http://promisesaplus.com/#point-41)
		 *
		 * @param {object} receiver `this` value for all handlers attached to
		 *  the returned promise.
		 * @returns {Promise}
		 */
		Promise.prototype['with'] = Promise.prototype.withThis = function(receiver) {
			var p = this._beget();
			var child = p._handler;
			child.receiver = receiver;
			this._handler.chain(child, receiver);
			return p;
		};

		return Promise;
	};

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));


},{}],22:[function(require,module,exports){
(function (process){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

/*global process,document,setTimeout,clearTimeout,MutationObserver,WebKitMutationObserver*/
(function(define) { 'use strict';
define(function(require) {
	/*jshint maxcomplexity:6*/

	// Sniff "best" async scheduling option
	// Prefer process.nextTick or MutationObserver, then check for
	// setTimeout, and finally vertx, since its the only env that doesn't
	// have setTimeout

	var MutationObs;
	var capturedSetTimeout = typeof setTimeout !== 'undefined' && setTimeout;

	// Default env
	var setTimer = function(f, ms) { return setTimeout(f, ms); };
	var clearTimer = function(t) { return clearTimeout(t); };
	var asap = function (f) { return capturedSetTimeout(f, 0); };

	// Detect specific env
	if (isNode()) { // Node
		asap = function (f) { return process.nextTick(f); };

	} else if (MutationObs = hasMutationObserver()) { // Modern browser
		asap = initMutationObserver(MutationObs);

	} else if (!capturedSetTimeout) { // vert.x
		var vertxRequire = require;
		var vertx = vertxRequire('vertx');
		setTimer = function (f, ms) { return vertx.setTimer(ms, f); };
		clearTimer = vertx.cancelTimer;
		asap = vertx.runOnLoop || vertx.runOnContext;
	}

	return {
		setTimer: setTimer,
		clearTimer: clearTimer,
		asap: asap
	};

	function isNode () {
		return typeof process !== 'undefined' &&
			Object.prototype.toString.call(process) === '[object process]';
	}

	function hasMutationObserver () {
	    return (typeof MutationObserver !== 'undefined' && MutationObserver) ||
			(typeof WebKitMutationObserver !== 'undefined' && WebKitMutationObserver);
	}

	function initMutationObserver(MutationObserver) {
		var scheduled;
		var node = document.createTextNode('');
		var o = new MutationObserver(run);
		o.observe(node, { characterData: true });

		function run() {
			var f = scheduled;
			scheduled = void 0;
			f();
		}

		var i = 0;
		return function (f) {
			scheduled = f;
			node.data = (i ^= 1);
		};
	}
});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(require); }));

}).call(this,require('_process'))
},{"_process":29}],23:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	return {
		formatError: formatError,
		formatObject: formatObject,
		tryStringify: tryStringify
	};

	/**
	 * Format an error into a string.  If e is an Error and has a stack property,
	 * it's returned.  Otherwise, e is formatted using formatObject, with a
	 * warning added about e not being a proper Error.
	 * @param {*} e
	 * @returns {String} formatted string, suitable for output to developers
	 */
	function formatError(e) {
		var s = typeof e === 'object' && e !== null && (e.stack || e.message) ? e.stack || e.message : formatObject(e);
		return e instanceof Error ? s : s + ' (WARNING: non-Error used)';
	}

	/**
	 * Format an object, detecting "plain" objects and running them through
	 * JSON.stringify if possible.
	 * @param {Object} o
	 * @returns {string}
	 */
	function formatObject(o) {
		var s = String(o);
		if(s === '[object Object]' && typeof JSON !== 'undefined') {
			s = tryStringify(o, s);
		}
		return s;
	}

	/**
	 * Try to return the result of JSON.stringify(x).  If that fails, return
	 * defaultValue
	 * @param {*} x
	 * @param {*} defaultValue
	 * @returns {String|*} JSON.stringify(x) or defaultValue
	 */
	function tryStringify(x, defaultValue) {
		try {
			return JSON.stringify(x);
		} catch(e) {
			return defaultValue;
		}
	}

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));

},{}],24:[function(require,module,exports){
(function (process){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	return function makePromise(environment) {

		var tasks = environment.scheduler;
		var emitRejection = initEmitRejection();

		var objectCreate = Object.create ||
			function(proto) {
				function Child() {}
				Child.prototype = proto;
				return new Child();
			};

		/**
		 * Create a promise whose fate is determined by resolver
		 * @constructor
		 * @returns {Promise} promise
		 * @name Promise
		 */
		function Promise(resolver, handler) {
			this._handler = resolver === Handler ? handler : init(resolver);
		}

		/**
		 * Run the supplied resolver
		 * @param resolver
		 * @returns {Pending}
		 */
		function init(resolver) {
			var handler = new Pending();

			try {
				resolver(promiseResolve, promiseReject, promiseNotify);
			} catch (e) {
				promiseReject(e);
			}

			return handler;

			/**
			 * Transition from pre-resolution state to post-resolution state, notifying
			 * all listeners of the ultimate fulfillment or rejection
			 * @param {*} x resolution value
			 */
			function promiseResolve (x) {
				handler.resolve(x);
			}
			/**
			 * Reject this promise with reason, which will be used verbatim
			 * @param {Error|*} reason rejection reason, strongly suggested
			 *   to be an Error type
			 */
			function promiseReject (reason) {
				handler.reject(reason);
			}

			/**
			 * @deprecated
			 * Issue a progress event, notifying all progress listeners
			 * @param {*} x progress event payload to pass to all listeners
			 */
			function promiseNotify (x) {
				handler.notify(x);
			}
		}

		// Creation

		Promise.resolve = resolve;
		Promise.reject = reject;
		Promise.never = never;

		Promise._defer = defer;
		Promise._handler = getHandler;

		/**
		 * Returns a trusted promise. If x is already a trusted promise, it is
		 * returned, otherwise returns a new trusted Promise which follows x.
		 * @param  {*} x
		 * @return {Promise} promise
		 */
		function resolve(x) {
			return isPromise(x) ? x
				: new Promise(Handler, new Async(getHandler(x)));
		}

		/**
		 * Return a reject promise with x as its reason (x is used verbatim)
		 * @param {*} x
		 * @returns {Promise} rejected promise
		 */
		function reject(x) {
			return new Promise(Handler, new Async(new Rejected(x)));
		}

		/**
		 * Return a promise that remains pending forever
		 * @returns {Promise} forever-pending promise.
		 */
		function never() {
			return foreverPendingPromise; // Should be frozen
		}

		/**
		 * Creates an internal {promise, resolver} pair
		 * @private
		 * @returns {Promise}
		 */
		function defer() {
			return new Promise(Handler, new Pending());
		}

		// Transformation and flow control

		/**
		 * Transform this promise's fulfillment value, returning a new Promise
		 * for the transformed result.  If the promise cannot be fulfilled, onRejected
		 * is called with the reason.  onProgress *may* be called with updates toward
		 * this promise's fulfillment.
		 * @param {function=} onFulfilled fulfillment handler
		 * @param {function=} onRejected rejection handler
		 * @param {function=} onProgress @deprecated progress handler
		 * @return {Promise} new promise
		 */
		Promise.prototype.then = function(onFulfilled, onRejected, onProgress) {
			var parent = this._handler;
			var state = parent.join().state();

			if ((typeof onFulfilled !== 'function' && state > 0) ||
				(typeof onRejected !== 'function' && state < 0)) {
				// Short circuit: value will not change, simply share handler
				return new this.constructor(Handler, parent);
			}

			var p = this._beget();
			var child = p._handler;

			parent.chain(child, parent.receiver, onFulfilled, onRejected, onProgress);

			return p;
		};

		/**
		 * If this promise cannot be fulfilled due to an error, call onRejected to
		 * handle the error. Shortcut for .then(undefined, onRejected)
		 * @param {function?} onRejected
		 * @return {Promise}
		 */
		Promise.prototype['catch'] = function(onRejected) {
			return this.then(void 0, onRejected);
		};

		/**
		 * Creates a new, pending promise of the same type as this promise
		 * @private
		 * @returns {Promise}
		 */
		Promise.prototype._beget = function() {
			return begetFrom(this._handler, this.constructor);
		};

		function begetFrom(parent, Promise) {
			var child = new Pending(parent.receiver, parent.join().context);
			return new Promise(Handler, child);
		}

		// Array combinators

		Promise.all = all;
		Promise.race = race;
		Promise._traverse = traverse;

		/**
		 * Return a promise that will fulfill when all promises in the
		 * input array have fulfilled, or will reject when one of the
		 * promises rejects.
		 * @param {array} promises array of promises
		 * @returns {Promise} promise for array of fulfillment values
		 */
		function all(promises) {
			return traverseWith(snd, null, promises);
		}

		/**
		 * Array<Promise<X>> -> Promise<Array<f(X)>>
		 * @private
		 * @param {function} f function to apply to each promise's value
		 * @param {Array} promises array of promises
		 * @returns {Promise} promise for transformed values
		 */
		function traverse(f, promises) {
			return traverseWith(tryCatch2, f, promises);
		}

		function traverseWith(tryMap, f, promises) {
			var handler = typeof f === 'function' ? mapAt : settleAt;

			var resolver = new Pending();
			var pending = promises.length >>> 0;
			var results = new Array(pending);

			for (var i = 0, x; i < promises.length && !resolver.resolved; ++i) {
				x = promises[i];

				if (x === void 0 && !(i in promises)) {
					--pending;
					continue;
				}

				traverseAt(promises, handler, i, x, resolver);
			}

			if(pending === 0) {
				resolver.become(new Fulfilled(results));
			}

			return new Promise(Handler, resolver);

			function mapAt(i, x, resolver) {
				if(!resolver.resolved) {
					traverseAt(promises, settleAt, i, tryMap(f, x, i), resolver);
				}
			}

			function settleAt(i, x, resolver) {
				results[i] = x;
				if(--pending === 0) {
					resolver.become(new Fulfilled(results));
				}
			}
		}

		function traverseAt(promises, handler, i, x, resolver) {
			if (maybeThenable(x)) {
				var h = getHandlerMaybeThenable(x);
				var s = h.state();

				if (s === 0) {
					h.fold(handler, i, void 0, resolver);
				} else if (s > 0) {
					handler(i, h.value, resolver);
				} else {
					resolver.become(h);
					visitRemaining(promises, i+1, h);
				}
			} else {
				handler(i, x, resolver);
			}
		}

		Promise._visitRemaining = visitRemaining;
		function visitRemaining(promises, start, handler) {
			for(var i=start; i<promises.length; ++i) {
				markAsHandled(getHandler(promises[i]), handler);
			}
		}

		function markAsHandled(h, handler) {
			if(h === handler) {
				return;
			}

			var s = h.state();
			if(s === 0) {
				h.visit(h, void 0, h._unreport);
			} else if(s < 0) {
				h._unreport();
			}
		}

		/**
		 * Fulfill-reject competitive race. Return a promise that will settle
		 * to the same state as the earliest input promise to settle.
		 *
		 * WARNING: The ES6 Promise spec requires that race()ing an empty array
		 * must return a promise that is pending forever.  This implementation
		 * returns a singleton forever-pending promise, the same singleton that is
		 * returned by Promise.never(), thus can be checked with ===
		 *
		 * @param {array} promises array of promises to race
		 * @returns {Promise} if input is non-empty, a promise that will settle
		 * to the same outcome as the earliest input promise to settle. if empty
		 * is empty, returns a promise that will never settle.
		 */
		function race(promises) {
			if(typeof promises !== 'object' || promises === null) {
				return reject(new TypeError('non-iterable passed to race()'));
			}

			// Sigh, race([]) is untestable unless we return *something*
			// that is recognizable without calling .then() on it.
			return promises.length === 0 ? never()
				 : promises.length === 1 ? resolve(promises[0])
				 : runRace(promises);
		}

		function runRace(promises) {
			var resolver = new Pending();
			var i, x, h;
			for(i=0; i<promises.length; ++i) {
				x = promises[i];
				if (x === void 0 && !(i in promises)) {
					continue;
				}

				h = getHandler(x);
				if(h.state() !== 0) {
					resolver.become(h);
					visitRemaining(promises, i+1, h);
					break;
				} else {
					h.visit(resolver, resolver.resolve, resolver.reject);
				}
			}
			return new Promise(Handler, resolver);
		}

		// Promise internals
		// Below this, everything is @private

		/**
		 * Get an appropriate handler for x, without checking for cycles
		 * @param {*} x
		 * @returns {object} handler
		 */
		function getHandler(x) {
			if(isPromise(x)) {
				return x._handler.join();
			}
			return maybeThenable(x) ? getHandlerUntrusted(x) : new Fulfilled(x);
		}

		/**
		 * Get a handler for thenable x.
		 * NOTE: You must only call this if maybeThenable(x) == true
		 * @param {object|function|Promise} x
		 * @returns {object} handler
		 */
		function getHandlerMaybeThenable(x) {
			return isPromise(x) ? x._handler.join() : getHandlerUntrusted(x);
		}

		/**
		 * Get a handler for potentially untrusted thenable x
		 * @param {*} x
		 * @returns {object} handler
		 */
		function getHandlerUntrusted(x) {
			try {
				var untrustedThen = x.then;
				return typeof untrustedThen === 'function'
					? new Thenable(untrustedThen, x)
					: new Fulfilled(x);
			} catch(e) {
				return new Rejected(e);
			}
		}

		/**
		 * Handler for a promise that is pending forever
		 * @constructor
		 */
		function Handler() {}

		Handler.prototype.when
			= Handler.prototype.become
			= Handler.prototype.notify // deprecated
			= Handler.prototype.fail
			= Handler.prototype._unreport
			= Handler.prototype._report
			= noop;

		Handler.prototype._state = 0;

		Handler.prototype.state = function() {
			return this._state;
		};

		/**
		 * Recursively collapse handler chain to find the handler
		 * nearest to the fully resolved value.
		 * @returns {object} handler nearest the fully resolved value
		 */
		Handler.prototype.join = function() {
			var h = this;
			while(h.handler !== void 0) {
				h = h.handler;
			}
			return h;
		};

		Handler.prototype.chain = function(to, receiver, fulfilled, rejected, progress) {
			this.when({
				resolver: to,
				receiver: receiver,
				fulfilled: fulfilled,
				rejected: rejected,
				progress: progress
			});
		};

		Handler.prototype.visit = function(receiver, fulfilled, rejected, progress) {
			this.chain(failIfRejected, receiver, fulfilled, rejected, progress);
		};

		Handler.prototype.fold = function(f, z, c, to) {
			this.when(new Fold(f, z, c, to));
		};

		/**
		 * Handler that invokes fail() on any handler it becomes
		 * @constructor
		 */
		function FailIfRejected() {}

		inherit(Handler, FailIfRejected);

		FailIfRejected.prototype.become = function(h) {
			h.fail();
		};

		var failIfRejected = new FailIfRejected();

		/**
		 * Handler that manages a queue of consumers waiting on a pending promise
		 * @constructor
		 */
		function Pending(receiver, inheritedContext) {
			Promise.createContext(this, inheritedContext);

			this.consumers = void 0;
			this.receiver = receiver;
			this.handler = void 0;
			this.resolved = false;
		}

		inherit(Handler, Pending);

		Pending.prototype._state = 0;

		Pending.prototype.resolve = function(x) {
			this.become(getHandler(x));
		};

		Pending.prototype.reject = function(x) {
			if(this.resolved) {
				return;
			}

			this.become(new Rejected(x));
		};

		Pending.prototype.join = function() {
			if (!this.resolved) {
				return this;
			}

			var h = this;

			while (h.handler !== void 0) {
				h = h.handler;
				if (h === this) {
					return this.handler = cycle();
				}
			}

			return h;
		};

		Pending.prototype.run = function() {
			var q = this.consumers;
			var handler = this.handler;
			this.handler = this.handler.join();
			this.consumers = void 0;

			for (var i = 0; i < q.length; ++i) {
				handler.when(q[i]);
			}
		};

		Pending.prototype.become = function(handler) {
			if(this.resolved) {
				return;
			}

			this.resolved = true;
			this.handler = handler;
			if(this.consumers !== void 0) {
				tasks.enqueue(this);
			}

			if(this.context !== void 0) {
				handler._report(this.context);
			}
		};

		Pending.prototype.when = function(continuation) {
			if(this.resolved) {
				tasks.enqueue(new ContinuationTask(continuation, this.handler));
			} else {
				if(this.consumers === void 0) {
					this.consumers = [continuation];
				} else {
					this.consumers.push(continuation);
				}
			}
		};

		/**
		 * @deprecated
		 */
		Pending.prototype.notify = function(x) {
			if(!this.resolved) {
				tasks.enqueue(new ProgressTask(x, this));
			}
		};

		Pending.prototype.fail = function(context) {
			var c = typeof context === 'undefined' ? this.context : context;
			this.resolved && this.handler.join().fail(c);
		};

		Pending.prototype._report = function(context) {
			this.resolved && this.handler.join()._report(context);
		};

		Pending.prototype._unreport = function() {
			this.resolved && this.handler.join()._unreport();
		};

		/**
		 * Wrap another handler and force it into a future stack
		 * @param {object} handler
		 * @constructor
		 */
		function Async(handler) {
			this.handler = handler;
		}

		inherit(Handler, Async);

		Async.prototype.when = function(continuation) {
			tasks.enqueue(new ContinuationTask(continuation, this));
		};

		Async.prototype._report = function(context) {
			this.join()._report(context);
		};

		Async.prototype._unreport = function() {
			this.join()._unreport();
		};

		/**
		 * Handler that wraps an untrusted thenable and assimilates it in a future stack
		 * @param {function} then
		 * @param {{then: function}} thenable
		 * @constructor
		 */
		function Thenable(then, thenable) {
			Pending.call(this);
			tasks.enqueue(new AssimilateTask(then, thenable, this));
		}

		inherit(Pending, Thenable);

		/**
		 * Handler for a fulfilled promise
		 * @param {*} x fulfillment value
		 * @constructor
		 */
		function Fulfilled(x) {
			Promise.createContext(this);
			this.value = x;
		}

		inherit(Handler, Fulfilled);

		Fulfilled.prototype._state = 1;

		Fulfilled.prototype.fold = function(f, z, c, to) {
			runContinuation3(f, z, this, c, to);
		};

		Fulfilled.prototype.when = function(cont) {
			runContinuation1(cont.fulfilled, this, cont.receiver, cont.resolver);
		};

		var errorId = 0;

		/**
		 * Handler for a rejected promise
		 * @param {*} x rejection reason
		 * @constructor
		 */
		function Rejected(x) {
			Promise.createContext(this);

			this.id = ++errorId;
			this.value = x;
			this.handled = false;
			this.reported = false;

			this._report();
		}

		inherit(Handler, Rejected);

		Rejected.prototype._state = -1;

		Rejected.prototype.fold = function(f, z, c, to) {
			to.become(this);
		};

		Rejected.prototype.when = function(cont) {
			if(typeof cont.rejected === 'function') {
				this._unreport();
			}
			runContinuation1(cont.rejected, this, cont.receiver, cont.resolver);
		};

		Rejected.prototype._report = function(context) {
			tasks.afterQueue(new ReportTask(this, context));
		};

		Rejected.prototype._unreport = function() {
			if(this.handled) {
				return;
			}
			this.handled = true;
			tasks.afterQueue(new UnreportTask(this));
		};

		Rejected.prototype.fail = function(context) {
			this.reported = true;
			emitRejection('unhandledRejection', this);
			Promise.onFatalRejection(this, context === void 0 ? this.context : context);
		};

		function ReportTask(rejection, context) {
			this.rejection = rejection;
			this.context = context;
		}

		ReportTask.prototype.run = function() {
			if(!this.rejection.handled && !this.rejection.reported) {
				this.rejection.reported = true;
				emitRejection('unhandledRejection', this.rejection) ||
					Promise.onPotentiallyUnhandledRejection(this.rejection, this.context);
			}
		};

		function UnreportTask(rejection) {
			this.rejection = rejection;
		}

		UnreportTask.prototype.run = function() {
			if(this.rejection.reported) {
				emitRejection('rejectionHandled', this.rejection) ||
					Promise.onPotentiallyUnhandledRejectionHandled(this.rejection);
			}
		};

		// Unhandled rejection hooks
		// By default, everything is a noop

		Promise.createContext
			= Promise.enterContext
			= Promise.exitContext
			= Promise.onPotentiallyUnhandledRejection
			= Promise.onPotentiallyUnhandledRejectionHandled
			= Promise.onFatalRejection
			= noop;

		// Errors and singletons

		var foreverPendingHandler = new Handler();
		var foreverPendingPromise = new Promise(Handler, foreverPendingHandler);

		function cycle() {
			return new Rejected(new TypeError('Promise cycle'));
		}

		// Task runners

		/**
		 * Run a single consumer
		 * @constructor
		 */
		function ContinuationTask(continuation, handler) {
			this.continuation = continuation;
			this.handler = handler;
		}

		ContinuationTask.prototype.run = function() {
			this.handler.join().when(this.continuation);
		};

		/**
		 * Run a queue of progress handlers
		 * @constructor
		 */
		function ProgressTask(value, handler) {
			this.handler = handler;
			this.value = value;
		}

		ProgressTask.prototype.run = function() {
			var q = this.handler.consumers;
			if(q === void 0) {
				return;
			}

			for (var c, i = 0; i < q.length; ++i) {
				c = q[i];
				runNotify(c.progress, this.value, this.handler, c.receiver, c.resolver);
			}
		};

		/**
		 * Assimilate a thenable, sending it's value to resolver
		 * @param {function} then
		 * @param {object|function} thenable
		 * @param {object} resolver
		 * @constructor
		 */
		function AssimilateTask(then, thenable, resolver) {
			this._then = then;
			this.thenable = thenable;
			this.resolver = resolver;
		}

		AssimilateTask.prototype.run = function() {
			var h = this.resolver;
			tryAssimilate(this._then, this.thenable, _resolve, _reject, _notify);

			function _resolve(x) { h.resolve(x); }
			function _reject(x)  { h.reject(x); }
			function _notify(x)  { h.notify(x); }
		};

		function tryAssimilate(then, thenable, resolve, reject, notify) {
			try {
				then.call(thenable, resolve, reject, notify);
			} catch (e) {
				reject(e);
			}
		}

		/**
		 * Fold a handler value with z
		 * @constructor
		 */
		function Fold(f, z, c, to) {
			this.f = f; this.z = z; this.c = c; this.to = to;
			this.resolver = failIfRejected;
			this.receiver = this;
		}

		Fold.prototype.fulfilled = function(x) {
			this.f.call(this.c, this.z, x, this.to);
		};

		Fold.prototype.rejected = function(x) {
			this.to.reject(x);
		};

		Fold.prototype.progress = function(x) {
			this.to.notify(x);
		};

		// Other helpers

		/**
		 * @param {*} x
		 * @returns {boolean} true iff x is a trusted Promise
		 */
		function isPromise(x) {
			return x instanceof Promise;
		}

		/**
		 * Test just enough to rule out primitives, in order to take faster
		 * paths in some code
		 * @param {*} x
		 * @returns {boolean} false iff x is guaranteed *not* to be a thenable
		 */
		function maybeThenable(x) {
			return (typeof x === 'object' || typeof x === 'function') && x !== null;
		}

		function runContinuation1(f, h, receiver, next) {
			if(typeof f !== 'function') {
				return next.become(h);
			}

			Promise.enterContext(h);
			tryCatchReject(f, h.value, receiver, next);
			Promise.exitContext();
		}

		function runContinuation3(f, x, h, receiver, next) {
			if(typeof f !== 'function') {
				return next.become(h);
			}

			Promise.enterContext(h);
			tryCatchReject3(f, x, h.value, receiver, next);
			Promise.exitContext();
		}

		/**
		 * @deprecated
		 */
		function runNotify(f, x, h, receiver, next) {
			if(typeof f !== 'function') {
				return next.notify(x);
			}

			Promise.enterContext(h);
			tryCatchReturn(f, x, receiver, next);
			Promise.exitContext();
		}

		function tryCatch2(f, a, b) {
			try {
				return f(a, b);
			} catch(e) {
				return reject(e);
			}
		}

		/**
		 * Return f.call(thisArg, x), or if it throws return a rejected promise for
		 * the thrown exception
		 */
		function tryCatchReject(f, x, thisArg, next) {
			try {
				next.become(getHandler(f.call(thisArg, x)));
			} catch(e) {
				next.become(new Rejected(e));
			}
		}

		/**
		 * Same as above, but includes the extra argument parameter.
		 */
		function tryCatchReject3(f, x, y, thisArg, next) {
			try {
				f.call(thisArg, x, y, next);
			} catch(e) {
				next.become(new Rejected(e));
			}
		}

		/**
		 * @deprecated
		 * Return f.call(thisArg, x), or if it throws, *return* the exception
		 */
		function tryCatchReturn(f, x, thisArg, next) {
			try {
				next.notify(f.call(thisArg, x));
			} catch(e) {
				next.notify(e);
			}
		}

		function inherit(Parent, Child) {
			Child.prototype = objectCreate(Parent.prototype);
			Child.prototype.constructor = Child;
		}

		function snd(x, y) {
			return y;
		}

		function noop() {}

		function hasCustomEvent() {
			if(typeof CustomEvent === 'function') {
				try {
					var ev = new CustomEvent('unhandledRejection');
					return ev instanceof CustomEvent;
				} catch (ignoredException) {}
			}
			return false;
		}

		function hasInternetExplorerCustomEvent() {
			if(typeof document !== 'undefined' && typeof document.createEvent === 'function') {
				try {
					// Try to create one event to make sure it's supported
					var ev = document.createEvent('CustomEvent');
					ev.initCustomEvent('eventType', false, true, {});
					return true;
				} catch (ignoredException) {}
			}
			return false;
		}

		function initEmitRejection() {
			/*global process, self, CustomEvent*/
			if(typeof process !== 'undefined' && process !== null
				&& typeof process.emit === 'function') {
				// Returning falsy here means to call the default
				// onPotentiallyUnhandledRejection API.  This is safe even in
				// browserify since process.emit always returns falsy in browserify:
				// https://github.com/defunctzombie/node-process/blob/master/browser.js#L40-L46
				return function(type, rejection) {
					return type === 'unhandledRejection'
						? process.emit(type, rejection.value, rejection)
						: process.emit(type, rejection);
				};
			} else if(typeof self !== 'undefined' && hasCustomEvent()) {
				return (function (self, CustomEvent) {
					return function (type, rejection) {
						var ev = new CustomEvent(type, {
							detail: {
								reason: rejection.value,
								key: rejection
							},
							bubbles: false,
							cancelable: true
						});

						return !self.dispatchEvent(ev);
					};
				}(self, CustomEvent));
			} else if(typeof self !== 'undefined' && hasInternetExplorerCustomEvent()) {
				return (function(self, document) {
					return function(type, rejection) {
						var ev = document.createEvent('CustomEvent');
						ev.initCustomEvent(type, false, true, {
							reason: rejection.value,
							key: rejection
						});

						return !self.dispatchEvent(ev);
					};
				}(self, document));
			}

			return noop;
		}

		return Promise;
	};
});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));

}).call(this,require('_process'))
},{"_process":29}],25:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */
/** @author Brian Cavalier */
/** @author John Hann */

(function(define) { 'use strict';
define(function() {

	return {
		pending: toPendingState,
		fulfilled: toFulfilledState,
		rejected: toRejectedState,
		inspect: inspect
	};

	function toPendingState() {
		return { state: 'pending' };
	}

	function toRejectedState(e) {
		return { state: 'rejected', reason: e };
	}

	function toFulfilledState(x) {
		return { state: 'fulfilled', value: x };
	}

	function inspect(handler) {
		var state = handler.state();
		return state === 0 ? toPendingState()
			 : state > 0   ? toFulfilledState(handler.value)
			               : toRejectedState(handler.value);
	}

});
}(typeof define === 'function' && define.amd ? define : function(factory) { module.exports = factory(); }));

},{}],26:[function(require,module,exports){
/** @license MIT License (c) copyright 2010-2014 original author or authors */

/**
 * Promises/A+ and when() implementation
 * when is part of the cujoJS family of libraries (http://cujojs.com/)
 * @author Brian Cavalier
 * @author John Hann
 */
(function(define) { 'use strict';
define(function (require) {

	var timed = require('./lib/decorators/timed');
	var array = require('./lib/decorators/array');
	var flow = require('./lib/decorators/flow');
	var fold = require('./lib/decorators/fold');
	var inspect = require('./lib/decorators/inspect');
	var generate = require('./lib/decorators/iterate');
	var progress = require('./lib/decorators/progress');
	var withThis = require('./lib/decorators/with');
	var unhandledRejection = require('./lib/decorators/unhandledRejection');
	var TimeoutError = require('./lib/TimeoutError');

	var Promise = [array, flow, fold, generate, progress,
		inspect, withThis, timed, unhandledRejection]
		.reduce(function(Promise, feature) {
			return feature(Promise);
		}, require('./lib/Promise'));

	var apply = require('./lib/apply')(Promise);

	// Public API

	when.promise     = promise;              // Create a pending promise
	when.resolve     = Promise.resolve;      // Create a resolved promise
	when.reject      = Promise.reject;       // Create a rejected promise

	when.lift        = lift;                 // lift a function to return promises
	when['try']      = attempt;              // call a function and return a promise
	when.attempt     = attempt;              // alias for when.try

	when.iterate     = Promise.iterate;      // DEPRECATED (use cujojs/most streams) Generate a stream of promises
	when.unfold      = Promise.unfold;       // DEPRECATED (use cujojs/most streams) Generate a stream of promises

	when.join        = join;                 // Join 2 or more promises

	when.all         = all;                  // Resolve a list of promises
	when.settle      = settle;               // Settle a list of promises

	when.any         = lift(Promise.any);    // One-winner race
	when.some        = lift(Promise.some);   // Multi-winner race
	when.race        = lift(Promise.race);   // First-to-settle race

	when.map         = map;                  // Array.map() for promises
	when.filter      = filter;               // Array.filter() for promises
	when.reduce      = lift(Promise.reduce);       // Array.reduce() for promises
	when.reduceRight = lift(Promise.reduceRight);  // Array.reduceRight() for promises

	when.isPromiseLike = isPromiseLike;      // Is something promise-like, aka thenable

	when.Promise     = Promise;              // Promise constructor
	when.defer       = defer;                // Create a {promise, resolve, reject} tuple

	// Error types

	when.TimeoutError = TimeoutError;

	/**
	 * Get a trusted promise for x, or by transforming x with onFulfilled
	 *
	 * @param {*} x
	 * @param {function?} onFulfilled callback to be called when x is
	 *   successfully fulfilled.  If promiseOrValue is an immediate value, callback
	 *   will be invoked immediately.
	 * @param {function?} onRejected callback to be called when x is
	 *   rejected.
	 * @param {function?} onProgress callback to be called when progress updates
	 *   are issued for x. @deprecated
	 * @returns {Promise} a new promise that will fulfill with the return
	 *   value of callback or errback or the completion value of promiseOrValue if
	 *   callback and/or errback is not supplied.
	 */
	function when(x, onFulfilled, onRejected, onProgress) {
		var p = Promise.resolve(x);
		if (arguments.length < 2) {
			return p;
		}

		return p.then(onFulfilled, onRejected, onProgress);
	}

	/**
	 * Creates a new promise whose fate is determined by resolver.
	 * @param {function} resolver function(resolve, reject, notify)
	 * @returns {Promise} promise whose fate is determine by resolver
	 */
	function promise(resolver) {
		return new Promise(resolver);
	}

	/**
	 * Lift the supplied function, creating a version of f that returns
	 * promises, and accepts promises as arguments.
	 * @param {function} f
	 * @returns {Function} version of f that returns promises
	 */
	function lift(f) {
		return function() {
			for(var i=0, l=arguments.length, a=new Array(l); i<l; ++i) {
				a[i] = arguments[i];
			}
			return apply(f, this, a);
		};
	}

	/**
	 * Call f in a future turn, with the supplied args, and return a promise
	 * for the result.
	 * @param {function} f
	 * @returns {Promise}
	 */
	function attempt(f /*, args... */) {
		/*jshint validthis:true */
		for(var i=0, l=arguments.length-1, a=new Array(l); i<l; ++i) {
			a[i] = arguments[i+1];
		}
		return apply(f, this, a);
	}

	/**
	 * Creates a {promise, resolver} pair, either or both of which
	 * may be given out safely to consumers.
	 * @return {{promise: Promise, resolve: function, reject: function, notify: function}}
	 */
	function defer() {
		return new Deferred();
	}

	function Deferred() {
		var p = Promise._defer();

		function resolve(x) { p._handler.resolve(x); }
		function reject(x) { p._handler.reject(x); }
		function notify(x) { p._handler.notify(x); }

		this.promise = p;
		this.resolve = resolve;
		this.reject = reject;
		this.notify = notify;
		this.resolver = { resolve: resolve, reject: reject, notify: notify };
	}

	/**
	 * Determines if x is promise-like, i.e. a thenable object
	 * NOTE: Will return true for *any thenable object*, and isn't truly
	 * safe, since it may attempt to access the `then` property of x (i.e.
	 *  clever/malicious getters may do weird things)
	 * @param {*} x anything
	 * @returns {boolean} true if x is promise-like
	 */
	function isPromiseLike(x) {
		return x && typeof x.then === 'function';
	}

	/**
	 * Return a promise that will resolve only once all the supplied arguments
	 * have resolved. The resolution value of the returned promise will be an array
	 * containing the resolution values of each of the arguments.
	 * @param {...*} arguments may be a mix of promises and values
	 * @returns {Promise}
	 */
	function join(/* ...promises */) {
		return Promise.all(arguments);
	}

	/**
	 * Return a promise that will fulfill once all input promises have
	 * fulfilled, or reject when any one input promise rejects.
	 * @param {array|Promise} promises array (or promise for an array) of promises
	 * @returns {Promise}
	 */
	function all(promises) {
		return when(promises, Promise.all);
	}

	/**
	 * Return a promise that will always fulfill with an array containing
	 * the outcome states of all input promises.  The returned promise
	 * will only reject if `promises` itself is a rejected promise.
	 * @param {array|Promise} promises array (or promise for an array) of promises
	 * @returns {Promise} promise for array of settled state descriptors
	 */
	function settle(promises) {
		return when(promises, Promise.settle);
	}

	/**
	 * Promise-aware array map function, similar to `Array.prototype.map()`,
	 * but input array may contain promises or values.
	 * @param {Array|Promise} promises array of anything, may contain promises and values
	 * @param {function(x:*, index:Number):*} mapFunc map function which may
	 *  return a promise or value
	 * @returns {Promise} promise that will fulfill with an array of mapped values
	 *  or reject if any input promise rejects.
	 */
	function map(promises, mapFunc) {
		return when(promises, function(promises) {
			return Promise.map(promises, mapFunc);
		});
	}

	/**
	 * Filter the provided array of promises using the provided predicate.  Input may
	 * contain promises and values
	 * @param {Array|Promise} promises array of promises and values
	 * @param {function(x:*, index:Number):boolean} predicate filtering predicate.
	 *  Must return truthy (or promise for truthy) for items to retain.
	 * @returns {Promise} promise that will fulfill with an array containing all items
	 *  for which predicate returned truthy.
	 */
	function filter(promises, predicate) {
		return when(promises, function(promises) {
			return Promise.filter(promises, predicate);
		});
	}

	return when;
});
})(typeof define === 'function' && define.amd ? define : function (factory) { module.exports = factory(require); });

},{"./lib/Promise":9,"./lib/TimeoutError":11,"./lib/apply":12,"./lib/decorators/array":13,"./lib/decorators/flow":14,"./lib/decorators/fold":15,"./lib/decorators/inspect":16,"./lib/decorators/iterate":17,"./lib/decorators/progress":18,"./lib/decorators/timed":19,"./lib/decorators/unhandledRejection":20,"./lib/decorators/with":21}],27:[function(require,module,exports){

},{}],28:[function(require,module,exports){
(function (process){
// .dirname, .basename, and .extname methods are extracted from Node.js v8.11.1,
// backported and transplited with Babel, with backwards-compat fixes

// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

// resolves . and .. elements in a path array with directory names there
// must be no slashes, empty elements, or device names (c:\) in the array
// (so also no leading and trailing slashes - it does not distinguish
// relative and absolute paths)
function normalizeArray(parts, allowAboveRoot) {
  // if the path tries to go above the root, `up` ends up > 0
  var up = 0;
  for (var i = parts.length - 1; i >= 0; i--) {
    var last = parts[i];
    if (last === '.') {
      parts.splice(i, 1);
    } else if (last === '..') {
      parts.splice(i, 1);
      up++;
    } else if (up) {
      parts.splice(i, 1);
      up--;
    }
  }

  // if the path is allowed to go above the root, restore leading ..s
  if (allowAboveRoot) {
    for (; up--; up) {
      parts.unshift('..');
    }
  }

  return parts;
}

// path.resolve([from ...], to)
// posix version
exports.resolve = function() {
  var resolvedPath = '',
      resolvedAbsolute = false;

  for (var i = arguments.length - 1; i >= -1 && !resolvedAbsolute; i--) {
    var path = (i >= 0) ? arguments[i] : process.cwd();

    // Skip empty and invalid entries
    if (typeof path !== 'string') {
      throw new TypeError('Arguments to path.resolve must be strings');
    } else if (!path) {
      continue;
    }

    resolvedPath = path + '/' + resolvedPath;
    resolvedAbsolute = path.charAt(0) === '/';
  }

  // At this point the path should be resolved to a full absolute path, but
  // handle relative paths to be safe (might happen when process.cwd() fails)

  // Normalize the path
  resolvedPath = normalizeArray(filter(resolvedPath.split('/'), function(p) {
    return !!p;
  }), !resolvedAbsolute).join('/');

  return ((resolvedAbsolute ? '/' : '') + resolvedPath) || '.';
};

// path.normalize(path)
// posix version
exports.normalize = function(path) {
  var isAbsolute = exports.isAbsolute(path),
      trailingSlash = substr(path, -1) === '/';

  // Normalize the path
  path = normalizeArray(filter(path.split('/'), function(p) {
    return !!p;
  }), !isAbsolute).join('/');

  if (!path && !isAbsolute) {
    path = '.';
  }
  if (path && trailingSlash) {
    path += '/';
  }

  return (isAbsolute ? '/' : '') + path;
};

// posix version
exports.isAbsolute = function(path) {
  return path.charAt(0) === '/';
};

// posix version
exports.join = function() {
  var paths = Array.prototype.slice.call(arguments, 0);
  return exports.normalize(filter(paths, function(p, index) {
    if (typeof p !== 'string') {
      throw new TypeError('Arguments to path.join must be strings');
    }
    return p;
  }).join('/'));
};


// path.relative(from, to)
// posix version
exports.relative = function(from, to) {
  from = exports.resolve(from).substr(1);
  to = exports.resolve(to).substr(1);

  function trim(arr) {
    var start = 0;
    for (; start < arr.length; start++) {
      if (arr[start] !== '') break;
    }

    var end = arr.length - 1;
    for (; end >= 0; end--) {
      if (arr[end] !== '') break;
    }

    if (start > end) return [];
    return arr.slice(start, end - start + 1);
  }

  var fromParts = trim(from.split('/'));
  var toParts = trim(to.split('/'));

  var length = Math.min(fromParts.length, toParts.length);
  var samePartsLength = length;
  for (var i = 0; i < length; i++) {
    if (fromParts[i] !== toParts[i]) {
      samePartsLength = i;
      break;
    }
  }

  var outputParts = [];
  for (var i = samePartsLength; i < fromParts.length; i++) {
    outputParts.push('..');
  }

  outputParts = outputParts.concat(toParts.slice(samePartsLength));

  return outputParts.join('/');
};

exports.sep = '/';
exports.delimiter = ':';

exports.dirname = function (path) {
  if (typeof path !== 'string') path = path + '';
  if (path.length === 0) return '.';
  var code = path.charCodeAt(0);
  var hasRoot = code === 47 /*/*/;
  var end = -1;
  var matchedSlash = true;
  for (var i = path.length - 1; i >= 1; --i) {
    code = path.charCodeAt(i);
    if (code === 47 /*/*/) {
        if (!matchedSlash) {
          end = i;
          break;
        }
      } else {
      // We saw the first non-path separator
      matchedSlash = false;
    }
  }

  if (end === -1) return hasRoot ? '/' : '.';
  if (hasRoot && end === 1) {
    // return '//';
    // Backwards-compat fix:
    return '/';
  }
  return path.slice(0, end);
};

function basename(path) {
  if (typeof path !== 'string') path = path + '';

  var start = 0;
  var end = -1;
  var matchedSlash = true;
  var i;

  for (i = path.length - 1; i >= 0; --i) {
    if (path.charCodeAt(i) === 47 /*/*/) {
        // If we reached a path separator that was not part of a set of path
        // separators at the end of the string, stop now
        if (!matchedSlash) {
          start = i + 1;
          break;
        }
      } else if (end === -1) {
      // We saw the first non-path separator, mark this as the end of our
      // path component
      matchedSlash = false;
      end = i + 1;
    }
  }

  if (end === -1) return '';
  return path.slice(start, end);
}

// Uses a mixed approach for backwards-compatibility, as ext behavior changed
// in new Node.js versions, so only basename() above is backported here
exports.basename = function (path, ext) {
  var f = basename(path);
  if (ext && f.substr(-1 * ext.length) === ext) {
    f = f.substr(0, f.length - ext.length);
  }
  return f;
};

exports.extname = function (path) {
  if (typeof path !== 'string') path = path + '';
  var startDot = -1;
  var startPart = 0;
  var end = -1;
  var matchedSlash = true;
  // Track the state of characters (if any) we see before our first dot and
  // after any path separator we find
  var preDotState = 0;
  for (var i = path.length - 1; i >= 0; --i) {
    var code = path.charCodeAt(i);
    if (code === 47 /*/*/) {
        // If we reached a path separator that was not part of a set of path
        // separators at the end of the string, stop now
        if (!matchedSlash) {
          startPart = i + 1;
          break;
        }
        continue;
      }
    if (end === -1) {
      // We saw the first non-path separator, mark this as the end of our
      // extension
      matchedSlash = false;
      end = i + 1;
    }
    if (code === 46 /*.*/) {
        // If this is our first dot, mark it as the start of our extension
        if (startDot === -1)
          startDot = i;
        else if (preDotState !== 1)
          preDotState = 1;
    } else if (startDot !== -1) {
      // We saw a non-dot and non-path separator before our dot, so we should
      // have a good chance at having a non-empty extension
      preDotState = -1;
    }
  }

  if (startDot === -1 || end === -1 ||
      // We saw a non-dot character immediately before the dot
      preDotState === 0 ||
      // The (right-most) trimmed path component is exactly '..'
      preDotState === 1 && startDot === end - 1 && startDot === startPart + 1) {
    return '';
  }
  return path.slice(startDot, end);
};

function filter (xs, f) {
    if (xs.filter) return xs.filter(f);
    var res = [];
    for (var i = 0; i < xs.length; i++) {
        if (f(xs[i], i, xs)) res.push(xs[i]);
    }
    return res;
}

// String.prototype.substr - negative index don't work in IE8
var substr = 'ab'.substr(-1) === 'b'
    ? function (str, start, len) { return str.substr(start, len) }
    : function (str, start, len) {
        if (start < 0) start = str.length + start;
        return str.substr(start, len);
    }
;

}).call(this,require('_process'))
},{"_process":29}],29:[function(require,module,exports){
// shim for using process in browser
var process = module.exports = {};

// cached from whatever global is present so that test runners that stub it
// don't break things.  But we need to wrap it in a try catch in case it is
// wrapped in strict mode code which doesn't define any globals.  It's inside a
// function because try/catches deoptimize in certain engines.

var cachedSetTimeout;
var cachedClearTimeout;

function defaultSetTimout() {
    throw new Error('setTimeout has not been defined');
}
function defaultClearTimeout () {
    throw new Error('clearTimeout has not been defined');
}
(function () {
    try {
        if (typeof setTimeout === 'function') {
            cachedSetTimeout = setTimeout;
        } else {
            cachedSetTimeout = defaultSetTimout;
        }
    } catch (e) {
        cachedSetTimeout = defaultSetTimout;
    }
    try {
        if (typeof clearTimeout === 'function') {
            cachedClearTimeout = clearTimeout;
        } else {
            cachedClearTimeout = defaultClearTimeout;
        }
    } catch (e) {
        cachedClearTimeout = defaultClearTimeout;
    }
} ())
function runTimeout(fun) {
    if (cachedSetTimeout === setTimeout) {
        //normal enviroments in sane situations
        return setTimeout(fun, 0);
    }
    // if setTimeout wasn't available but was latter defined
    if ((cachedSetTimeout === defaultSetTimout || !cachedSetTimeout) && setTimeout) {
        cachedSetTimeout = setTimeout;
        return setTimeout(fun, 0);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedSetTimeout(fun, 0);
    } catch(e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't trust the global object when called normally
            return cachedSetTimeout.call(null, fun, 0);
        } catch(e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error
            return cachedSetTimeout.call(this, fun, 0);
        }
    }


}
function runClearTimeout(marker) {
    if (cachedClearTimeout === clearTimeout) {
        //normal enviroments in sane situations
        return clearTimeout(marker);
    }
    // if clearTimeout wasn't available but was latter defined
    if ((cachedClearTimeout === defaultClearTimeout || !cachedClearTimeout) && clearTimeout) {
        cachedClearTimeout = clearTimeout;
        return clearTimeout(marker);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedClearTimeout(marker);
    } catch (e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't  trust the global object when called normally
            return cachedClearTimeout.call(null, marker);
        } catch (e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error.
            // Some versions of I.E. have different rules for clearTimeout vs setTimeout
            return cachedClearTimeout.call(this, marker);
        }
    }



}
var queue = [];
var draining = false;
var currentQueue;
var queueIndex = -1;

function cleanUpNextTick() {
    if (!draining || !currentQueue) {
        return;
    }
    draining = false;
    if (currentQueue.length) {
        queue = currentQueue.concat(queue);
    } else {
        queueIndex = -1;
    }
    if (queue.length) {
        drainQueue();
    }
}

function drainQueue() {
    if (draining) {
        return;
    }
    var timeout = runTimeout(cleanUpNextTick);
    draining = true;

    var len = queue.length;
    while(len) {
        currentQueue = queue;
        queue = [];
        while (++queueIndex < len) {
            if (currentQueue) {
                currentQueue[queueIndex].run();
            }
        }
        queueIndex = -1;
        len = queue.length;
    }
    currentQueue = null;
    draining = false;
    runClearTimeout(timeout);
}

process.nextTick = function (fun) {
    var args = new Array(arguments.length - 1);
    if (arguments.length > 1) {
        for (var i = 1; i < arguments.length; i++) {
            args[i - 1] = arguments[i];
        }
    }
    queue.push(new Item(fun, args));
    if (queue.length === 1 && !draining) {
        runTimeout(drainQueue);
    }
};

// v8 likes predictible objects
function Item(fun, array) {
    this.fun = fun;
    this.array = array;
}
Item.prototype.run = function () {
    this.fun.apply(null, this.array);
};
process.title = 'browser';
process.browser = true;
process.env = {};
process.argv = [];
process.version = ''; // empty string to avoid regexp issues
process.versions = {};

function noop() {}

process.on = noop;
process.addListener = noop;
process.once = noop;
process.off = noop;
process.removeListener = noop;
process.removeAllListeners = noop;
process.emit = noop;
process.prependListener = noop;
process.prependOnceListener = noop;

process.listeners = function (name) { return [] }

process.binding = function (name) {
    throw new Error('process.binding is not supported');
};

process.cwd = function () { return '/' };
process.chdir = function (dir) {
    throw new Error('process.chdir is not supported');
};
process.umask = function() { return 0; };

},{}],30:[function(require,module,exports){
if (typeof Object.create === 'function') {
  // implementation from standard node.js 'util' module
  module.exports = function inherits(ctor, superCtor) {
    ctor.super_ = superCtor
    ctor.prototype = Object.create(superCtor.prototype, {
      constructor: {
        value: ctor,
        enumerable: false,
        writable: true,
        configurable: true
      }
    });
  };
} else {
  // old school shim for old browsers
  module.exports = function inherits(ctor, superCtor) {
    ctor.super_ = superCtor
    var TempCtor = function () {}
    TempCtor.prototype = superCtor.prototype
    ctor.prototype = new TempCtor()
    ctor.prototype.constructor = ctor
  }
}

},{}],31:[function(require,module,exports){
module.exports = function isBuffer(arg) {
  return arg && typeof arg === 'object'
    && typeof arg.copy === 'function'
    && typeof arg.fill === 'function'
    && typeof arg.readUInt8 === 'function';
}
},{}],32:[function(require,module,exports){
(function (process,global){
// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

var formatRegExp = /%[sdj%]/g;
exports.format = function(f) {
  if (!isString(f)) {
    var objects = [];
    for (var i = 0; i < arguments.length; i++) {
      objects.push(inspect(arguments[i]));
    }
    return objects.join(' ');
  }

  var i = 1;
  var args = arguments;
  var len = args.length;
  var str = String(f).replace(formatRegExp, function(x) {
    if (x === '%%') return '%';
    if (i >= len) return x;
    switch (x) {
      case '%s': return String(args[i++]);
      case '%d': return Number(args[i++]);
      case '%j':
        try {
          return JSON.stringify(args[i++]);
        } catch (_) {
          return '[Circular]';
        }
      default:
        return x;
    }
  });
  for (var x = args[i]; i < len; x = args[++i]) {
    if (isNull(x) || !isObject(x)) {
      str += ' ' + x;
    } else {
      str += ' ' + inspect(x);
    }
  }
  return str;
};


// Mark that a method should not be used.
// Returns a modified function which warns once by default.
// If --no-deprecation is set, then it is a no-op.
exports.deprecate = function(fn, msg) {
  // Allow for deprecating things in the process of starting up.
  if (isUndefined(global.process)) {
    return function() {
      return exports.deprecate(fn, msg).apply(this, arguments);
    };
  }

  if (process.noDeprecation === true) {
    return fn;
  }

  var warned = false;
  function deprecated() {
    if (!warned) {
      if (process.throwDeprecation) {
        throw new Error(msg);
      } else if (process.traceDeprecation) {
        console.trace(msg);
      } else {
        console.error(msg);
      }
      warned = true;
    }
    return fn.apply(this, arguments);
  }

  return deprecated;
};


var debugs = {};
var debugEnviron;
exports.debuglog = function(set) {
  if (isUndefined(debugEnviron))
    debugEnviron = process.env.NODE_DEBUG || '';
  set = set.toUpperCase();
  if (!debugs[set]) {
    if (new RegExp('\\b' + set + '\\b', 'i').test(debugEnviron)) {
      var pid = process.pid;
      debugs[set] = function() {
        var msg = exports.format.apply(exports, arguments);
        console.error('%s %d: %s', set, pid, msg);
      };
    } else {
      debugs[set] = function() {};
    }
  }
  return debugs[set];
};


/**
 * Echos the value of a value. Trys to print the value out
 * in the best way possible given the different types.
 *
 * @param {Object} obj The object to print out.
 * @param {Object} opts Optional options object that alters the output.
 */
/* legacy: obj, showHidden, depth, colors*/
function inspect(obj, opts) {
  // default options
  var ctx = {
    seen: [],
    stylize: stylizeNoColor
  };
  // legacy...
  if (arguments.length >= 3) ctx.depth = arguments[2];
  if (arguments.length >= 4) ctx.colors = arguments[3];
  if (isBoolean(opts)) {
    // legacy...
    ctx.showHidden = opts;
  } else if (opts) {
    // got an "options" object
    exports._extend(ctx, opts);
  }
  // set default options
  if (isUndefined(ctx.showHidden)) ctx.showHidden = false;
  if (isUndefined(ctx.depth)) ctx.depth = 2;
  if (isUndefined(ctx.colors)) ctx.colors = false;
  if (isUndefined(ctx.customInspect)) ctx.customInspect = true;
  if (ctx.colors) ctx.stylize = stylizeWithColor;
  return formatValue(ctx, obj, ctx.depth);
}
exports.inspect = inspect;


// http://en.wikipedia.org/wiki/ANSI_escape_code#graphics
inspect.colors = {
  'bold' : [1, 22],
  'italic' : [3, 23],
  'underline' : [4, 24],
  'inverse' : [7, 27],
  'white' : [37, 39],
  'grey' : [90, 39],
  'black' : [30, 39],
  'blue' : [34, 39],
  'cyan' : [36, 39],
  'green' : [32, 39],
  'magenta' : [35, 39],
  'red' : [31, 39],
  'yellow' : [33, 39]
};

// Don't use 'blue' not visible on cmd.exe
inspect.styles = {
  'special': 'cyan',
  'number': 'yellow',
  'boolean': 'yellow',
  'undefined': 'grey',
  'null': 'bold',
  'string': 'green',
  'date': 'magenta',
  // "name": intentionally not styling
  'regexp': 'red'
};


function stylizeWithColor(str, styleType) {
  var style = inspect.styles[styleType];

  if (style) {
    return '\u001b[' + inspect.colors[style][0] + 'm' + str +
           '\u001b[' + inspect.colors[style][1] + 'm';
  } else {
    return str;
  }
}


function stylizeNoColor(str, styleType) {
  return str;
}


function arrayToHash(array) {
  var hash = {};

  array.forEach(function(val, idx) {
    hash[val] = true;
  });

  return hash;
}


function formatValue(ctx, value, recurseTimes) {
  // Provide a hook for user-specified inspect functions.
  // Check that value is an object with an inspect function on it
  if (ctx.customInspect &&
      value &&
      isFunction(value.inspect) &&
      // Filter out the util module, it's inspect function is special
      value.inspect !== exports.inspect &&
      // Also filter out any prototype objects using the circular check.
      !(value.constructor && value.constructor.prototype === value)) {
    var ret = value.inspect(recurseTimes, ctx);
    if (!isString(ret)) {
      ret = formatValue(ctx, ret, recurseTimes);
    }
    return ret;
  }

  // Primitive types cannot have properties
  var primitive = formatPrimitive(ctx, value);
  if (primitive) {
    return primitive;
  }

  // Look up the keys of the object.
  var keys = Object.keys(value);
  var visibleKeys = arrayToHash(keys);

  if (ctx.showHidden) {
    keys = Object.getOwnPropertyNames(value);
  }

  // IE doesn't make error fields non-enumerable
  // http://msdn.microsoft.com/en-us/library/ie/dww52sbt(v=vs.94).aspx
  if (isError(value)
      && (keys.indexOf('message') >= 0 || keys.indexOf('description') >= 0)) {
    return formatError(value);
  }

  // Some type of object without properties can be shortcutted.
  if (keys.length === 0) {
    if (isFunction(value)) {
      var name = value.name ? ': ' + value.name : '';
      return ctx.stylize('[Function' + name + ']', 'special');
    }
    if (isRegExp(value)) {
      return ctx.stylize(RegExp.prototype.toString.call(value), 'regexp');
    }
    if (isDate(value)) {
      return ctx.stylize(Date.prototype.toString.call(value), 'date');
    }
    if (isError(value)) {
      return formatError(value);
    }
  }

  var base = '', array = false, braces = ['{', '}'];

  // Make Array say that they are Array
  if (isArray(value)) {
    array = true;
    braces = ['[', ']'];
  }

  // Make functions say that they are functions
  if (isFunction(value)) {
    var n = value.name ? ': ' + value.name : '';
    base = ' [Function' + n + ']';
  }

  // Make RegExps say that they are RegExps
  if (isRegExp(value)) {
    base = ' ' + RegExp.prototype.toString.call(value);
  }

  // Make dates with properties first say the date
  if (isDate(value)) {
    base = ' ' + Date.prototype.toUTCString.call(value);
  }

  // Make error with message first say the error
  if (isError(value)) {
    base = ' ' + formatError(value);
  }

  if (keys.length === 0 && (!array || value.length == 0)) {
    return braces[0] + base + braces[1];
  }

  if (recurseTimes < 0) {
    if (isRegExp(value)) {
      return ctx.stylize(RegExp.prototype.toString.call(value), 'regexp');
    } else {
      return ctx.stylize('[Object]', 'special');
    }
  }

  ctx.seen.push(value);

  var output;
  if (array) {
    output = formatArray(ctx, value, recurseTimes, visibleKeys, keys);
  } else {
    output = keys.map(function(key) {
      return formatProperty(ctx, value, recurseTimes, visibleKeys, key, array);
    });
  }

  ctx.seen.pop();

  return reduceToSingleString(output, base, braces);
}


function formatPrimitive(ctx, value) {
  if (isUndefined(value))
    return ctx.stylize('undefined', 'undefined');
  if (isString(value)) {
    var simple = '\'' + JSON.stringify(value).replace(/^"|"$/g, '')
                                             .replace(/'/g, "\\'")
                                             .replace(/\\"/g, '"') + '\'';
    return ctx.stylize(simple, 'string');
  }
  if (isNumber(value))
    return ctx.stylize('' + value, 'number');
  if (isBoolean(value))
    return ctx.stylize('' + value, 'boolean');
  // For some reason typeof null is "object", so special case here.
  if (isNull(value))
    return ctx.stylize('null', 'null');
}


function formatError(value) {
  return '[' + Error.prototype.toString.call(value) + ']';
}


function formatArray(ctx, value, recurseTimes, visibleKeys, keys) {
  var output = [];
  for (var i = 0, l = value.length; i < l; ++i) {
    if (hasOwnProperty(value, String(i))) {
      output.push(formatProperty(ctx, value, recurseTimes, visibleKeys,
          String(i), true));
    } else {
      output.push('');
    }
  }
  keys.forEach(function(key) {
    if (!key.match(/^\d+$/)) {
      output.push(formatProperty(ctx, value, recurseTimes, visibleKeys,
          key, true));
    }
  });
  return output;
}


function formatProperty(ctx, value, recurseTimes, visibleKeys, key, array) {
  var name, str, desc;
  desc = Object.getOwnPropertyDescriptor(value, key) || { value: value[key] };
  if (desc.get) {
    if (desc.set) {
      str = ctx.stylize('[Getter/Setter]', 'special');
    } else {
      str = ctx.stylize('[Getter]', 'special');
    }
  } else {
    if (desc.set) {
      str = ctx.stylize('[Setter]', 'special');
    }
  }
  if (!hasOwnProperty(visibleKeys, key)) {
    name = '[' + key + ']';
  }
  if (!str) {
    if (ctx.seen.indexOf(desc.value) < 0) {
      if (isNull(recurseTimes)) {
        str = formatValue(ctx, desc.value, null);
      } else {
        str = formatValue(ctx, desc.value, recurseTimes - 1);
      }
      if (str.indexOf('\n') > -1) {
        if (array) {
          str = str.split('\n').map(function(line) {
            return '  ' + line;
          }).join('\n').substr(2);
        } else {
          str = '\n' + str.split('\n').map(function(line) {
            return '   ' + line;
          }).join('\n');
        }
      }
    } else {
      str = ctx.stylize('[Circular]', 'special');
    }
  }
  if (isUndefined(name)) {
    if (array && key.match(/^\d+$/)) {
      return str;
    }
    name = JSON.stringify('' + key);
    if (name.match(/^"([a-zA-Z_][a-zA-Z_0-9]*)"$/)) {
      name = name.substr(1, name.length - 2);
      name = ctx.stylize(name, 'name');
    } else {
      name = name.replace(/'/g, "\\'")
                 .replace(/\\"/g, '"')
                 .replace(/(^"|"$)/g, "'");
      name = ctx.stylize(name, 'string');
    }
  }

  return name + ': ' + str;
}


function reduceToSingleString(output, base, braces) {
  var numLinesEst = 0;
  var length = output.reduce(function(prev, cur) {
    numLinesEst++;
    if (cur.indexOf('\n') >= 0) numLinesEst++;
    return prev + cur.replace(/\u001b\[\d\d?m/g, '').length + 1;
  }, 0);

  if (length > 60) {
    return braces[0] +
           (base === '' ? '' : base + '\n ') +
           ' ' +
           output.join(',\n  ') +
           ' ' +
           braces[1];
  }

  return braces[0] + base + ' ' + output.join(', ') + ' ' + braces[1];
}


// NOTE: These type checking functions intentionally don't use `instanceof`
// because it is fragile and can be easily faked with `Object.create()`.
function isArray(ar) {
  return Array.isArray(ar);
}
exports.isArray = isArray;

function isBoolean(arg) {
  return typeof arg === 'boolean';
}
exports.isBoolean = isBoolean;

function isNull(arg) {
  return arg === null;
}
exports.isNull = isNull;

function isNullOrUndefined(arg) {
  return arg == null;
}
exports.isNullOrUndefined = isNullOrUndefined;

function isNumber(arg) {
  return typeof arg === 'number';
}
exports.isNumber = isNumber;

function isString(arg) {
  return typeof arg === 'string';
}
exports.isString = isString;

function isSymbol(arg) {
  return typeof arg === 'symbol';
}
exports.isSymbol = isSymbol;

function isUndefined(arg) {
  return arg === void 0;
}
exports.isUndefined = isUndefined;

function isRegExp(re) {
  return isObject(re) && objectToString(re) === '[object RegExp]';
}
exports.isRegExp = isRegExp;

function isObject(arg) {
  return typeof arg === 'object' && arg !== null;
}
exports.isObject = isObject;

function isDate(d) {
  return isObject(d) && objectToString(d) === '[object Date]';
}
exports.isDate = isDate;

function isError(e) {
  return isObject(e) &&
      (objectToString(e) === '[object Error]' || e instanceof Error);
}
exports.isError = isError;

function isFunction(arg) {
  return typeof arg === 'function';
}
exports.isFunction = isFunction;

function isPrimitive(arg) {
  return arg === null ||
         typeof arg === 'boolean' ||
         typeof arg === 'number' ||
         typeof arg === 'string' ||
         typeof arg === 'symbol' ||  // ES6 symbol
         typeof arg === 'undefined';
}
exports.isPrimitive = isPrimitive;

exports.isBuffer = require('./support/isBuffer');

function objectToString(o) {
  return Object.prototype.toString.call(o);
}


function pad(n) {
  return n < 10 ? '0' + n.toString(10) : n.toString(10);
}


var months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
              'Oct', 'Nov', 'Dec'];

// 26 Feb 16:19:34
function timestamp() {
  var d = new Date();
  var time = [pad(d.getHours()),
              pad(d.getMinutes()),
              pad(d.getSeconds())].join(':');
  return [d.getDate(), months[d.getMonth()], time].join(' ');
}


// log is just a thin wrapper to console.log that prepends a timestamp
exports.log = function() {
  console.log('%s - %s', timestamp(), exports.format.apply(exports, arguments));
};


/**
 * Inherit the prototype methods from one constructor into another.
 *
 * The Function.prototype.inherits from lang.js rewritten as a standalone
 * function (not on Function.prototype). NOTE: If this file is to be loaded
 * during bootstrapping this function needs to be rewritten using some native
 * functions as prototype setup using normal JavaScript does not work as
 * expected during bootstrapping (see mirror.js in r114903).
 *
 * @param {function} ctor Constructor function which needs to inherit the
 *     prototype.
 * @param {function} superCtor Constructor function to inherit prototype from.
 */
exports.inherits = require('inherits');

exports._extend = function(origin, add) {
  // Don't do anything if add isn't an object
  if (!add || !isObject(add)) return origin;

  var keys = Object.keys(add);
  var i = keys.length;
  while (i--) {
    origin[keys[i]] = add[keys[i]];
  }
  return origin;
};

function hasOwnProperty(obj, prop) {
  return Object.prototype.hasOwnProperty.call(obj, prop);
}

}).call(this,require('_process'),typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./support/isBuffer":31,"_process":29,"inherits":30}]},{},[1]);
