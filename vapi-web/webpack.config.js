// webpack.config.js
const path = require("path");

module.exports = {
  entry: "./index.js",
  output: {
    filename: "vapi.min.js",
    path: path.resolve(__dirname, "dist"),
    library: "Vapi",
    libraryTarget: "umd",
    libraryExport: "default", 
    globalObject: "this",
  },
  mode: "production",
};
