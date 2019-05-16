'use strict'

var gulp = require('gulp'),
  scss = require('gulp-sass'),
  sourcemaps = require('gulp-sourcemaps'),
  babel = require('gulp-babel'),
  concat = require('gulp-concat'),
  plumber = require('gulp-plumber'),
  imagemin = require('gulp-imagemin')

var SASS_INCLUDE_PATHS = [
  './node_modules/normalize-scss/sass/'
]

function handleError (err) {
  console.log(err.toString())
  this.emit('end')
}

gulp.task('styles', function () {
  return gulp.src('./src/sass/main.scss')
    .pipe(plumber({errorHandler: handleError}))
    .pipe(sourcemaps.init())
    .pipe(scss({outputStyle: 'compressed', includePaths: SASS_INCLUDE_PATHS}))
    .pipe(sourcemaps.write())
    .pipe(gulp.dest('./static/css'))
})

gulp.task('js', function () {
  return gulp.src('./src/js/**/*.js')
    .pipe(plumber({errorHandler: handleError}))
    .pipe(sourcemaps.init())
    .pipe(babel({compact: true}))
    .pipe(concat('main.js'))
    .pipe(sourcemaps.write())
    .pipe(gulp.dest('./static/js'))
})

gulp.task('images', function () {
  return gulp.src('./src/img/**/*')
    .pipe(imagemin())
    .pipe(gulp.dest('./static/img'))
})

gulp.task('watch', ['styles', 'js'], function () {
  gulp.watch('./src/sass/**/*.scss', ['styles'])
  gulp.watch('./src/js/**/*.js', ['js'])
})

gulp.task('default', ['styles', 'js', 'images'], function () {

})