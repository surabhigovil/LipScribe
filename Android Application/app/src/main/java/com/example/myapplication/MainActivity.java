package com.example.myapplication;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;

import com.chaquo.python.PyObject;
import com.google.android.material.snackbar.Snackbar;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.util.Log;
import android.view.View;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentTransaction;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.myapplication.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;

public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;
    private final int VIDEO_RECORD_CODE=101;
    private static int CAMERA_PERMISSION_CODE = 100;
    ActivityResultLauncher<Intent> activityResultLauncher;
    private Uri videoPath;
    final LoadingDialog loadingDialog = new LoadingDialog(MainActivity.this);


    TextView textView;
//    private ProgressBar spinner;

    private void recordVideo(){
        Intent intent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
        activityResultLauncher.launch(intent);
    }
    public void captureVideo(View view)
    {
        recordVideo();
    }

    private boolean isCameraPresentInPhone(){
        if(getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            return true;
        }
            else
            {
                return false;
            }
        }
    private void getCameraPermission(){
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        }
    }
    private void getStoragePermission(){
        Log.i("Storage","storage permission");
        if(ContextCompat.checkSelfPermission(this, String.valueOf(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE}))
                == PackageManager.PERMISSION_DENIED){
            Log.i("storage","permission not given");
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.READ_EXTERNAL_STORAGE},200);
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        appBarConfiguration = new AppBarConfiguration.Builder(navController.getGraph()).build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);

//        Button b1 = (Button) findViewById(R.id.button_first);
//        b1.setVisibility(View.INVISIBLE);

//        spinner = findViewById(R.id.progressBar_cyclic);


        binding.fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
//                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
//                        .setAction("Action", null).show();
//                loadingDialog.startLoadingDialog();
            }
        });
//        getStoragePermission();
        if(isCameraPresentInPhone())
        {
            Log.i("VIDEO_RECORD_TAG", "Camera is Detected");
            getCameraPermission();

        }
        else
        {
            Log.i("VIDEO_RECORD_TAG", "No Camera is not Detected");
        }
        activityResultLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult result) {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    videoPath = result.getData().getData();
                    Log.i("Video_URI", "URI new is "+videoPath);
                    String path= getPath(videoPath);
                    new GetPathTask(path).execute();

                }
            }
        });
    }
    private final class GetPathTask extends AsyncTask<Void, Void, String> {
        private String path;
        public GetPathTask(String path) {
            this.path = path;
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
//            spinner.setVisibility(View.VISIBLE);
            loadingDialog.startLoadingDialog();
        }

        @Override
        protected String doInBackground(Void... params) {
            String word=executePython(path);
            return word;
        }

        @Override
        protected void onPostExecute(String word) {
//            spinner.setVisibility(View.INVISIBLE);
            loadingDialog.dismissDialog();
            Log.i("Predicted word is", "Word new  is "+word);

            //opening second fragment and sending word to it
            View myView = findViewById(R.id.button_first);
            myView.performClick();

            SecondFragment secFragment = new SecondFragment();
            FragmentTransaction fragmentTransaction = getSupportFragmentManager().beginTransaction();
            Bundle data = new Bundle();
            data.putString("word",word);
            secFragment.setArguments(data);
            fragmentTransaction.replace(R.id.nav_host_fragment_content_main,secFragment).commit();

            Log.i("Inside second fragment", "Word new  is "+word);

            File fdelete = new File(videoPath.getPath());
            if (fdelete.exists()) {
                if (fdelete.delete()) {
                    Log.i("file Deleted :", videoPath.getPath());
                } else {
                    Log.i("file not Deleted :", videoPath.getPath());
                }
            }
        }

    }
    public String executePython(String video_path)
    {
//        ProgressBar progressBar_cyclic;
//        spinner = findViewById(R.id.progressBar_cyclic);
//        spinner.setVisibility(View.VISIBLE);
        if(!Python.isStarted())
        {
            Python.start(new AndroidPlatform(this));
        }
        Python py = Python.getInstance();
        Log.i("video_path","video path is "+video_path);
//        PyObject pyObj = py.getModule("helloWorld");
//        PyObject obj = pyObj.callAttr("main");
        Log.i("video_path","video path is before getting module "+video_path);
        PyObject pyObj = py.getModule("preprocess");
        Log.i("video_path","video path is after getting module "+video_path);
        PyObject obj = pyObj.callAttr("video_to_npy_array",video_path);


        Log.i("python test", "msg is"+obj.toString());
        return obj.toString();

    }

    public String getPath(Uri uri) {
        Cursor cursor = null;
        try {
            String[] projection = {MediaStore.Images.Media.DATA};
            cursor = getContentResolver().query(uri, projection, null, null, null);

            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            return cursor.getString(column_index);
        }finally {
            if (cursor != null) {
                cursor.close();
            }
        }
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }
}