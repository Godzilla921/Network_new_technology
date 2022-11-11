package com.example.helloworld

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val btn = findViewById<Button>(R.id.button)
        val text =findViewById<TextView>(R.id.textView)
        btn.setOnClickListener {
            text.text = "Hello World! This is Jin Yage!"
            text.visibility=TextView.VISIBLE
        }
    }
}