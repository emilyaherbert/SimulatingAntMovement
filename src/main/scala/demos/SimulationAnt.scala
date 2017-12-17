package demos

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j
import scalafx.scene.canvas.GraphicsContext
import scalafx.scene.paint.Color
import utility.ModelInfo
import utility.Model

class SimulationAnt(var cx: Double,
    var cy: Double,
    speed: Double,
    model: Model,
    var modelInfo: ModelInfo) {

  def move(delta: Double) {
    modelInfo = model.makePrediction(modelInfo)
    cx += modelInfo.vx * speed * delta
    cy += modelInfo.vy * speed * delta
  }

  def display(gc: GraphicsContext) {
    gc.fill = Color.White
    gc.fillOval(cx - 5, cy - 5, 10, 10)
  }

  /*
  def save(pw: PixelWriter) {
    for (i <- (0 max (cx.toInt - 5)) until (600 min (cx.toInt + 5))) {
      for (j <- (0 max (cy.toInt - 5)) until (600 min (cy.toInt + 5))) {
        pw.setColor(i, j, Color.White)
      }
    }
  }
  * 
  */

  private def updateArray(arr: Array[Double], x: Double, y: Double): Array[Double] = {
    for (i <- 2 until arr.length) {
      arr(i - 2) = arr(i)
    }
    arr(arr.length - 2) = x
    arr(arr.length - 1) = y
    arr
  }
}