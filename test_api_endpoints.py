import requests
import json

BASE_URL = "http://localhost:8000"

def test_image_upload(image_path):
    """Test uploading an image for analysis"""
    print(f"Testing image upload: {image_path}")
    
    try:
        # Open and prepare image
        with open(image_path, 'rb') as f:
            files = {'image': (image_path, f, 'image/png')}
            
            # Send request
            response = requests.post(
                f"{BASE_URL}/analyze",
                files=files,
                timeout=30
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if result["success"]:
                print("✅ Analysis Successful!")
                analysis = result["analysis"]
                
                # Display skin analysis
                print(f"\n📊 Skin Analysis Results:")
                print(f"   Monk Skin Tone: {analysis['monk_skin_tone']['prediction']}")
                print(f"   Confidence: {analysis['monk_skin_tone']['confidence']:.2%}")
                print(f"   Fitzpatrick: {analysis['fitzpatrick']['prediction']}")
                print(f"   Confidence: {analysis['fitzpatrick']['confidence']:.2%}")
                
                # Get recommendations based on analysis
                print("\n🛍️ Getting Recommendations...")
                rec_response = requests.post(
                    f"{BASE_URL}/recommend",
                    params={
                        "monk_skin_tone": analysis['monk_skin_tone']['prediction'],
                        "fitzpatrick": analysis['fitzpatrick']['prediction'],
                        "budget": 50,
                        "n_recommendations": 5
                    }
                )
                
                if rec_response.status_code == 200:
                    rec_result = rec_response.json()
                    if rec_result["success"]:
                        print(f"✅ Found {rec_result['total_recommendations']} recommendations")
                        print("\nTop Recommendations:")
                        for i, product in enumerate(rec_result["recommendations"], 1):
                            print(f"{i}. {product['name']}")
                            print(f"   💰 ${product['price']:.2f} | ⭐ {product['rating']}/5")
                            print(f"   🎯 Match: {product['match_score']}/1.0")
                            print()
                    else:
                        print("❌ Failed to get recommendations")
                else:
                    print("❌ Recommendation request failed")
                
                return analysis
            else:
                print(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

def test_with_direct_params():
    """Test with direct skin parameters (no image)"""
    print("\n🔧 Testing Direct Parameters...")
    
    params = {
        "monk_skin_tone": 4,
        "fitzpatrick": 5,
        "budget": 75,
        "n_recommendations": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/recommend", params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print(f"✅ Recommendations for Monk {params['monk_skin_tone']}, Fitzpatrick {params['fitzpatrick']}:")
                for i, product in enumerate(result["recommendations"], 1):
                    print(f"{i}. {product['name']} (${product['price']:.2f})")
            else:
                print(f"❌ Failed: {result}")
        else:
            print(f"❌ Request failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_all_endpoints():
    """Test all API endpoints"""
    print("="*60)
    print("BEAUTY RECOMMENDATION SYSTEM - API TEST")
    print("="*60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: Browse products
    print("\n2. Testing product browser...")
    try:
        response = requests.get(f"{BASE_URL}/products", params={"page_size": 3}, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print(f"   ✅ Found {result['total_products']} products total")
                print(f"   Showing {len(result['products'])} products")
            else:
                print(f"   ❌ Failed: {result}")
        else:
            print(f"   ❌ Status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Test with your image
    print("\n3. Testing image analysis...")
    image_path = r"C:\Users\HP\Desktop\remboglow model\dataset\images\-10814178347722472.png"
    
    # Check if image exists
    import os
    if os.path.exists(image_path):
        print(f"   Found test image: {image_path}")
        test_image_upload(image_path)
    else:
        print(f"   ❌ Test image not found: {image_path}")
        print("   Testing with direct parameters instead...")
        test_with_direct_params()
    
    print("\n" + "="*60)
    print("🎉 API TEST COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    test_all_endpoints()