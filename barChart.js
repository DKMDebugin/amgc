const d3 = require('d3'); // reference d3 library

// get dataset
const url = 'GDP-data.json';
const dataset = fetch(url).then(response => response.text())
              .then(json=>JSON.parse(json)).then((dataset)=> dataset)
              .catch((e)=>{console.log(e.message)});

const w = 800
const h = 500;
const barSpace = 20;
const padding = 60;

// assign info from dataset
const data = dataset.then(dataset => dataset.data);

data.then(function(data) {
  const barWidth = h / data.length;
  let date = data.map(item => new Date(item[0]));
  let xMax = new Date(d3.max(date))
  let gdp = data.map(item => item[1]);

  // chart scale
  let xScale = d3.scaleTime()
                  .domain([d3.min(date), xMax])
                  .range([padding, w]);
  // let xScale = d3.scaleLinear()
  //                 .domain([0, d3.max(data, d => new Date(d[0]))])
  //                 .range([padding, w - padding]);
  let yScale = d3.scaleLinear()
                  .domain([0, d3.max(gdp)])
                  .range([padding, h - padding]);
  let yAxisScale = d3.scaleLinear()
                  .domain([0, d3.max(gdp)])
                  .range([h - padding,  padding]);


  // set svg width, height & class attribute
  let svg = d3.select('svg')
              .attr('width', w)
              .attr('height', h)
              .attr('class', 'svg-container');

  // create bars dynamically
  svg.selectAll('rect')
    .data(data.map(i => yScale(i))) // data seeding
    .enter() // data binding
    .append('rect') // create rect
    .attr('x', (d, i) => xScale(date[i]))
    .attr('y', (d, i) => h - d)
    .attr('height', (d, i) => d)
    .attr('width', d => barWidth)
    // .attr('transform', (d, i) => {
    //   let translate = [barWidth * i + padding, -padding];
    //   return `translate(${translate})`;
    // })
    .attr('class', 'bar')
    .attr('data-date', d => d[0])
    .attr('data-gdp', d => d[1])

  // draw  & attach axes
  const xAxis = d3.axisBottom(xScale);
  const yAxis = d3.axisLeft(yAxisScale);

  svg.append('g')
      .attr('transform', `translate(0, ${h - padding})`)
      .attr('id', 'x-axis')
      .call(xAxis);
  svg.append('g')
      .attr('transform', `translate(${padding}, 0)`)
      .attr('id', 'y-axis')
      .call(yAxis);
});
// // chart scale
// let xScale = d3.scaleLinear()
//                 .domain([0, d3.max(data.then(data=>data), d => d[0].slice(0,4))])
//                 .range([0, w - padding]);
// let yScale = d3.scaleLinear()
//                 .domain([0, d3.max(data.then(data=>data), d => d[1])])
//                 .range([h - padding, padding]);
//
//
// // set svg attr
// let svg = d3.select('svg')
//             .attr('width', w)
//             .attr('height', h)
//             .attr('class', 'svg-container');
//
// // create bars dynamically
// // svg.selectAll('rect')
// //   .data(data.then(data=>data)) // data seeding
// //   .enter()
// //   .append('rect')
// //   .attr('x', d => h - xScale(barWidth + padding))
// //   .attr('y', d => h - yScale(d[1]))
// //   .attr('height', d => d[1])
// //   .attr('width', d => xScale(barWidth))
// //   .attr('class', 'bar')
// svg.selectAll('rect')
//   .data(data.then(data=>data)) // data seeding
//   .enter()
//   .append('rect')
//   .attr('x', (d, i) => 30 * i)
//   .attr('y', 450)
//   .attr('height', 100)
//   .attr('width', 20)
//   .attr('class', 'bar')
//
// // draw  & attach axes
// const xAxis = d3.axisBottom(xScale);
// const yAxis = d3.axisLeft(yScale);
// svg.append('g')
//     .attr('transform', `translate(0, ${h - padding})`)
//     .attr('id', 'x-axis')
//     .call(xAxis);
// svg.append('g')
//     .attr('transform', `translate(${padding - 30}, 0)`)
//     .attr('id', 'y-axis')
//     .call(yAxis);
