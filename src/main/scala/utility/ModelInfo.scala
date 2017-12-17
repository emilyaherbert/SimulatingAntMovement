package utility

abstract class ModelInfo(var vx: Double, var vy: Double) {
  def update(nvx: Double, nvy: Double): ModelInfo
}

object NormalModelInfo {
  def apply():NormalModelInfo = {
    new NormalModelInfo(Math.random() * 2 - 1.0,Math.random() * 2 - 1.0)
  }
}

class NormalModelInfo(_vx: Double, _vy: Double) extends ModelInfo(_vx, _vy) {

  def update(nvx: Double, nvy: Double): NormalModelInfo = {
    vx = nvx
    vy = nvy
    this
  }
}