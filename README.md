# CÁC PHƯƠNG PHÁP TỐI ƯU HÓA TRONG HUẤN LUYỆN MÔ HÌNH HỌC MÁY
## Gradient Descent
Gradient Descent là một phương pháp quan trọng trong tối ưu hóa mô hình máy học. Gradient Descent sử dụng đạo hàm để xác định hướng và độ lớn của bước di chuyển để giảm thiểu giá trị của hàm mất mát.
### Nguyên lí
Nguyên lý cơ bản của Gradient Descent (GD) là một phương pháp tối ưu hóa được sử dụng để điều chỉnh các tham số của mô hình máy học sao cho giảm thiểu hàm mất mát (loss function).
Các bước thực hiện phương pháp Gradient Descent: 
Bước 1: Khởi tạo tham số: Khởi tạo các tham số của mô hình, chẳng hạn như trọng số và bias.
Bước 2: Chọn hyperparameters (tham số): Chọn giá trị cho các hyperparameters như learning rate (α), là một tham số quan trọng quyết định độ lớn của bước cập nhật.
Bước 3: Tính Gradient: Tính toán gradient của hàm mất mát theo các tham số.
∇J=[ ∂J/∂θ1 , ∂J/∂θ2 ,…, ∂J/∂θn]
Bước 4: Cập nhật tham số: Cập nhật các tham số của mô hình bằng cách di chuyển ngược hướng của gradient với một bước có kích thước là learning rate:  
∇θ = θ − α⋅∇J
Bước 5: Lặp Lại Quá Trình: Lặp lại quá trình tính toán và cập nhật cho đến khi đạt được điều kiện dừng (ví dụ: số lần lặp, giảm gradient đủ nhỏ, hoặc đạt được giá trị mất mát mong muốn).
### Ưu-nhược điểm và dữ liệu thích hợp
Ưu Điểm:
	Đơn giản: Dễ hiểu và triển khai.
	Phổ biến: Là một trong những phương pháp tối ưu hóa phổ biến và được sử dụng nhiều.
Nhược Điểm:
	Chọn kích thước bước khó khăn: Việc chọn kích thước bước (learning rate) phù hợp có thể là một thách thức.
	Khả năng hội tụ trong các trường hợp phức tạp chậm: Trong một số trường hợp, có thể yêu cầu nhiều thời gian để hội tụ đến điểm tối ưu.
Dữ liệu phù hợp: Phi tuyến tính, hội tụ nhanh chóng đến giá trị tối ưu, không thay đổi quá nhanh theo thời gian, có nhiều biến đầu vào.
## Batch Gradient Descent
### Nguyên lí
Batch Gradient Descent điều chỉnh tham số của mô hình bằng cách sử dụng thông tin từ toàn bộ tập dữ liệu huấn luyện trong mỗi lần cập nhật. Mục tiêu là hội tụ đến giá trị tối ưu của hàm mất mát, nơi mô hình có thể dự đoán tốt trên dữ liệu mới.
Các bước thực hiện Batch Gradient Descent:
Bước 1: Khởi tạo tham số: Khởi tạo các tham số của mô hình, chẳng hạn như trọng số và bias.
Bước 2: Chọn hyperparameters: Chọn giá trị cho các hyperparameters như learning rate (α), là một tham số quan trọng quyết định độ lớn của bước cập nhật.
Bước 3: Lặp qua toàn bộ tập huấn luyện mỗi lần cập nhật tham số. Điều này bao gồm tính toán gradient cho toàn bộ dữ liệu.
Bước 4: Tính Gradient: Tính toán gradient của hàm mất mát theo các tham số.
∇J=[ ∂J/∂θ1 , ∂J/∂θ2 ,…, ∂J/∂θn]
Bước 5: Cập nhật tham số: Cập nhật các tham số của mô hình bằng cách di chuyển ngược hướng của gradient với một bước có kích thước là learning rate:  
∇θ = θ − α⋅∇J
Bước 6: Lặp lại quá trình: Lặp lại quá trình bước 3-5 và cập nhật cho đến khi đạt được điều kiện dừng (ví dụ: số lần lặp, giảm gradient đủ nhỏ, hoặc đạt được giá trị mất mát mong muốn).
### Ưu-nhược điểm và dữ liệu thích hợp
Ưu điểm:
	Độ chính xác cao: Batch Gradient Descent thường dẫn đến độ chính xác cao hơn vì nó sử dụng toàn bộ tập dữ liệu để cập nhật tham số.
	Hội tụ ổn định: Do sử dụng toàn bộ tập dữ liệu, Batch Gradient Descent có thể hội tụ một cách ổn định đến giá trị tối ưu.
	Hiệu quả trên dữ liệu nhỏ: Trong trường hợp dữ liệu nhỏ có thể fit vào bộ nhớ, Batch Gradient Descent có thể hoạt động hiệu quả.
	Được ưa chuộng trong học sâu: Trong các mô hình học sâu, Batch Gradient Descent thường được ưa chuộng do có thể tận dụng khả năng tính toán của GPU để xử lý toàn bộ tập dữ liệu.
Nhược Điểm:
	Tính tính toán cao: Batch Gradient Descent yêu cầu tính toán đạo hàm của toàn bộ tập dữ liệu, điều này có thể làm chậm quá trình đào tạo đối với dữ liệu lớn.
	Không hiệu quả cho dữ liệu lớn: Khi tập dữ liệu không thể fit vào bộ nhớ, Batch Gradient Descent trở nên không hiệu quả vì đòi hỏi nhiều tài nguyên tính toán.
	Khả năng mất thông tin với dữ liệu động: Nếu dữ liệu thay đổi theo thời gian, Batch Gradient Descent có thể mất đi thông tin về biến động nếu không được cập nhật thường xuyên..
	Yêu cầu bộ nhớ lớn: Cần phải lưu toàn bộ tập dữ liệu trong bộ nhớ, điều này có thể trở thành vấn đề khi làm việc với dữ liệu lớn.
Dữ liệu phù hợp đối với phương pháp: dữ liệu nhỏ và sạch, không nhiễu và không chứa quá nhiều giá trị ngoại lệ, không hoặc ít thay đổi theo thời gian.
## Stochastic Gradient Descent (SGD)
### Nguyên lí
Mục tiêu của SGD là tối ưu hóa hàm mất mát, biểu diễn sự chênh lệch giữa dự đoán của mô hình và giá trị thực tế trên tập dữ liệu huấn luyện. Thay vì tính gradient trên toàn bộ tập dữ liệu, SGD giảm độ phức tạp tính toán bằng cách chỉ sử dụng một lượng nhỏ dữ liệu trong mỗi lần cập nhật. Quá trình này được lặp lại qua nhiều mini-batch cho đến khi đạt được điều kiện dừng.
Các bước thực hiện SGD:
Bước 1: Khởi tạo tham số: Khởi tạo các tham số của mô hình, bao gồm trọng số và độ chệch (bias).
Bước 2: Chọn kích thước mẫu (batch size): Chọn một số lượng mẫu ngẫu nhiên từ tập dữ liệu để tạo thành một batch. Kích thước batch là một tham số quan trọng trong SGD.
Bước 3: Tính giá trị dự đoán: Sử dụng các tham số hiện tại của mô hình để tính toán giá trị dự đoán cho mỗi mẫu trong batch.
Bước 4: Tính toán hàm mất mát: Tính toán giá trị của hàm mất mát, đo lường sự chênh lệch giữa giá trị dự đoán và giá trị thực tế.
Bước 5: Tính gradient của hàm mất mát: Tính toán gradient của hàm mất mát đối với từng tham số. Điều này đại diện cho hướng và độ lớn cần điều chỉnh các tham số để giảm mất mát.
Bước 6: Cập nhật tham số: Sử dụng gradient tính được để cập nhật các tham số của mô hình. Công thức cập nhật thường sử dụng một tỷ lệ học (learning rate) để kiểm soát kích thước của bước cập nhật.
Tham số mới=Tham số cũ −Tỉ lệ học×Gradient
Bước 7: Lặp lại quá trình: Lặp lại các bước 2-6 cho đến khi đạt được điều kiện dừng hoặc một số lượng vòng lặp đã đủ.
### Ưu-nhược điểm và dữ liệu thích hợp
Ưu điểm:
	Hiệu quả cho dữ liệu lớn: SGD thích hợp cho việc đào tạo mô hình trên dữ liệu lớn, vì nó chỉ yêu cầu một lượng nhỏ dữ liệu trong mỗi lần cập nhật.
	Khả năng đối phó với dữ liệu nhiễu: Do việc chọn ngẫu nhiên mini-batch, SGD có khả năng tránh được nhiễu và giúp mô hình chống lại điểm cực tiểu địa phương.
	Tích hợp tốt với học sâu: Trong học sâu, SGD được sử dụng phổ biến giúp đào tạo mô hình nhanh chóng.
	Cập nhật thường xuyên: Việc cập nhật tham số sau mỗi mini-batch giúp SGD có thể hội tụ nhanh chóng và thích hợp cho việc đào tạo online.
	Ngốn ít tài nguyên tính toán: Vì chỉ sử dụng một lượng nhỏ dữ liệu, SGD ngốn ít tài nguyên tính toán hơn so với Batch Gradient Descent.
Nhược Điểm:
	Khả năng dao động: SGD có thể dao động quanh điểm tối ưu do sự ngẫu nhiên trong việc chọn mini-batch.
	Không đảm bảo hội tụ đến giá trị tối ưu toàn cục: Do tính ngẫu nhiên, SGD không đảm bảo hội tụ đến giá trị tối ưu toàn cục, nhưng thay vào đó có thể dẫn đến điểm cực tiểu địa phương.
	Yêu cầu chọn learning rate thích hợp: Cần phải chọn learning rate phù hợp để đảm bảo sự hội tụ và tránh trường hợp overshooting hoặc undershooting.
	Dễ bị ảnh hưởng bởi nhiễu: Do sự ngẫu nhiên trong việc chọn mini-batch, SGD có thể bị ảnh hưởng bởi nhiễu, đặc biệt là trên dữ liệu nhỏ hoặc có nhiễu.
	Không hiệu quả đối với dữ liệu nhỏ: Trong trường hợp dữ liệu nhỏ, SGD có thể không hiệu quả do độ dao động cao và không tận dụng được các lợi ích của Batch Gradient Descent.
Dữ liệu thích hợp: lớn, dữ liệu động, có thể thay đổi theo thời gian, có thể có nhiễu, dữ liệu yêu cầu huấn luyện nhanh hoặc huấn luyện online liên tục được cập nhật với dữ liệu mới.
## RMSprop (Root Mean Square Propagation)
### Nguyên lí
Nguyên lý cơ bản của RMSprop là điều chỉnh tỷ lệ học (learning rate) cho từng tham số của mô hình dựa trên lịch sử của các gradient gần đây. Cụ thể, RMSprop sử dụng trung bình bình phương của các gradient để thay đổi tỷ lệ học.
Các bước thực hiện RMSprop:
Bước 1: Khởi tạo các tham số: Khởi tạo các tham số của mô hình, bao gồm các trọng số (θ) và các tham số khác.
Bước 2: Khởi tạo các biến cho RMSprop: Khởi tạo biến E[g2] lưu trữ trung bình bình phương của gradient.
Bước 3: Thiết lập các hyperparameters: Đặt giá trị cho các siêu tham số như tỷ lệ học (η) và hệ số giảm trọng số (β).
Bước 4: Lặp lại qua từng mini-batch: Tính gradient (∇J) của hàm mất mát đối với các tham số.
Bước 5: Cập nhật E[g2] Sử dụng công thức sau để cập nhật trung bình bình phương của gradient: E[g2]=βE[g2]+(1−β)(∇J)2
Bước 6: Cập nhật tham sốSử dụng công thức sau để cập nhật các tham số: 
θ=θ -  η /(E[g2]+ϵ).∇J 
ϵ lúc này là một giá trị nhỏ để tránh chia cho 0.
Bước 7: Lặp lại  quá trình: Lặp lại các bước trên với các mini-batch khác hoặc toàn bộ dữ liệu.
### Ưu nhược điểm và dữ liệu thích hợp
Ưu điểm:
	Điều chỉnh tỷ lệ học tự động: RMSprop có khả năng tự động điều chỉnh tỷ lệ học cho từng tham số của mô hình dựa trên lịch sử của gradient. Điều này giúp nó thích ứng tốt với các tình huống trong đó gradient thay đổi đáng kể.
	Hiệu suất tốt trong các bài toán không đồng nhất: RMSprop thường hoạt động hiệu quả hơn so với các thuật toán tối ưu hóa khác trong các bài toán có hàm mất mát không đồng nhất hoặc có các tham số có độ biến động lớn.
	Không yêu cầu tham số điều chỉnh nhiều: RMSprop ít yêu cầu sự điều chỉnh tham số hơn so với một số thuật toán khác
Nhược điểm:
	Nguy cơ đột ngột giảm tỷ lệ học: Trong một số trường hợp, RMSprop có thể dẫn đến nguy cơ tỷ lệ học giảm đột ngột quá nhanh, đặc biệt khi giá trị của E[g2] là rất lớn. Điều này có thể làm chậm quá trình học.
	Không hiệu quả trên tất cả các bài toán: Mặc dù RMSprop thích ứng tốt với nhiều tình huống, nhưng nó không phải là giải pháp hoàn hảo cho mọi bài toán. Có những trường hợp nó không hiệu quả hoặc thậm chí dẫn đến hiện tượng không ổn định.
	Tùy chỉnh các siêu tham số: Mặc dù ít cần điều chỉnh hơn so với một số thuật toán khác, nhưng vẫn có một số siêu tham số cần được điều chỉnh, như tỷ lệ học (η) và hệ số giảm trọng số (β).
Dữ liệu thích hợp: thường là những bài toán mà hàm mất mát có tính chất không đồng nhất, nghĩa là gradient thay đổi đáng kể tại các điểm khác nhau của không gian tham số; dữ liệu trong các bài toán học tăng cường; xử lí ảnh và thị giác máy tính.
## Momentum
### Nguyên lí
Nguyên lý cơ bản của thuật toán Momentum là sử dụng một đối tượng giữ đà (momentum) để theo dõi hướng và tốc độ của quá trình tối ưu hóa.
Bước 1: Khởi tạo các tham số: Khởi tạo các tham số của mô hình, bao gồm các trọng số (θ) và các tham số khác.
Bước 2: Khởi tạo moment (động lượng): Khởi tạo moment bậc nhất (m) bằng 0 hoặc giá trị khác nhau tùy thuộc vào chiến lược khởi tạo.
Bước 3: Thiết lập tỷ lệ học và hệ số giảm trọng số: Đặt giá trị cho tỷ lệ học (η) và hệ số giảm trọng số (β).
Bước 4: Lặp qua từng mini-batch hoặc toàn bộ dữ liệu: Tính gradient (∇J) của hàm mất mát đối với các tham số.
Bước 5: Cập nhật moment: Sử dụng công thức sau để cập nhật moment: m=βm+(1−β)∇J
Bước 6: Cập nhật tham số: Sử dụng công thức sau để cập nhật các tham số: θ=θ−ηm
Bước 7: Lặp lại quá trình: Lặp lại các bước trên với các mini-batch khác hoặc toàn bộ dữ liệu.
### Ưu nhược điểm và dữ liệu thích hợp
Ưu điểm:
	Tăng tốc độ học: Momentum giúp tăng tốc độ học của mô hình bằng cách giữ đà (momentum) từ các bước trước đó, giúp vượt qua sự đánh mất độ đạo hàm và giảm bớt độ dao động trong quá trình tối ưu hóa.
	Khả năng vượt qua điểm cực tiểu cục bộ: Đối với các bài toán có nhiều điểm cực tiểu cục bộ, Momentum có khả năng vượt qua chúng và tiếp tục tìm kiếm trong không gian tham số.
	Giảm dao động: Momentum giúp giảm độ dao động của quá trình tối ưu hóa, giúp mô hình hội tụ nhanh hơn.
Nhược điểm:
	Quá trình học có thể quá nhanh: Trong một số trường hợp, Momentum có thể làm cho quá trình học tiến triển quá nhanh và vượt qua điểm tối ưu.
	Khả năng lặp đi lặp lại: Nếu tỷ lệ học quá lớn, mô hình có thể lặp đi lặp lại quanh điểm tối ưu hoặc không hội tụ.
	Yêu cầu lựa chọn tham số tốt: Cần lựa chọn tham số β một cách cẩn thận để đảm bảo sự ổn định và hiệu suất tốt của thuật toán.
Dữ liệu phù hợp cho Momentum là các dạng dữ liệu có đặc tính không đồng nhất, dữ liệu cho bài toán có độ biến động lớn.
## Adam (Adaptive Moment Estimation)
### Nguyên lí
Adam kết hợp hai ý tưởng chính từ RMSprop và Momentum để tạo ra một phương pháp hiệu quả cho việc tối ưu hóa hàm mất mát.
Các bước thực hiện Adam:
Bước 1: Khởi tạo các tham số: Khởi tạo các tham số của mô hình, bao gồm các trọng số (θ) và các tham số khác.
Bước 2: Khởi tạo moment bậc nhất và bậc hai: Khởi tạo moment bậc nhất (m) và moment bậc hai (v) bằng 0 hoặc giá trị khác nhau tùy thuộc vào chiến lược khởi tạo.
Bước 3: Thiết lập tỷ lệ học và các hệ số giảm trọng số: Đặt giá trị cho tỷ lệ học (η), hệ số giảm trọng số cho moment bậc nhất (β1), và hệ số giảm trọng số cho moment bậc hai (β2).
Bước 4: Khởi tạo biến đếm:
Khởi tạo biến đếm (t) bằng 0.
Bước 5: Lặp qua từng mini-batch hoặc toàn bộ dữ liệu:
Tính gradient (∇J) của hàm mất mát đối với các tham số.
Bước 6: Cập nhật moment bậc nhất và bậc hai: Sử dụng công thức sau để cập nhật moment bậc nhất và moment bậc hai: 
m=β1m+(1−β1)∇J 
v=β2v+(1−β2)(∇J)2
Bước 7: Hiệu chỉnh độ chệch (bias correction):
Thực hiện bước hiệu chỉnh độ chệch để giảm ảnh hưởng của giá trị khởi tạo 0 cho m ̂ và v ̂
m ̂=m/(1−β1t) 
v ̂=v/(1−β2t)
Bước 8: Cập nhật tham số:
Sử dụng công thức sau để cập nhật các tham số 
θ = θ − η/(√(v ̂ )+ϵm ̂) 
Trong đó:
ϵ là một giá trị nhỏ để tránh chia cho 0.
Bước 9: Cập nhật biến đếm:
Tăng giá trị biến đếm (t) lên 1: 1t=t+1.
Bước 10: Lặp lại quá trình:
Lặp lại các bước trên với các mini-batch khác hoặc toàn bộ dữ liệu.
### Ưu nhược điểm và dữ liệu thích hợp 
Ưu điểm:
	Hiệu suất và linh hoạt: Adam thường hiệu quả và linh hoạt trên nhiều loại bài toán học máy, đặc biệt là trong các mô hình sâu và phức tạp.
	Tự điều chỉnh tỷ lệ học: Adam tự điều chỉnh tỷ lệ học (learning rate) cho từng tham số dựa trên lịch sử của gradient, giúp nó phù hợp với nhiều tình huống và giảm nguy cơ tỷ lệ học giảm quá nhanh.
	Hoạt động tốt với dữ liệu lớn: Adam thường hoạt động tốt với các tập dữ liệu lớn và có đặc tính không đồng nhất.
Nhược điểm:
	Yêu cầu đối số tinh chỉnh: Adam có nhiều đối số cần được tinh chỉnh, bao gồm β1, β2, và ϵ. Sự lựa chọn không tốt có thể dẫn đến hiện tượng không ổn định hoặc quá mức tinh chỉnh.
	Nhạy cảm với outlier: Adam có thể nhạy cảm với các giá trị gradient ngoại lệ (outlier) và có thể bị ảnh hưởng nếu các gradient có biên độ lớn.
Dữ liệu thích hợp: lớn và không đồng nhất, nơi tỷ lệ học có thể được điều chỉnh linh hoạt cho từng tham số; dữ liệu cho các bài toán tối ưu hóa phức tạp.
## Adagrad (Adaptive Gradient Algorithm)
### Nguyên lí
Nguyên lý cơ bản của Adagrad là điều chỉnh tỷ lệ học dựa trên lịch sử của gradient cho mỗi tham số.
Các bước thực hiện Adagrad:
Bước 1: Khởi tạo các tham số: Khởi tạo các tham số của mô hình, bao gồm các trọng số (θ) và các tham số khác.
Bước 2: Khởi tạo ma trận tổng bình phương gradient: Khởi tạo ma trận tổng bình phương gradient (G) bằng 0 hoặc giá trị khác nhau tùy thuộc vào chiến lược khởi tạo.
Bước 3: Thiết lập tỷ lệ học và giá trị epsilon: Đặt giá trị cho tỷ lệ học (η) và giá trị epsilon (ϵ).
Bước 4: Lặp qua từng mini-batch hoặc toàn bộ dữ liệu: Tính gradient (∇J) của hàm mất mát đối với từng tham số của mô hình.
Bước 5: Cập nhật tổng bình phương gradient: Sử dụng công thức sau để cập nhật ma trận tổng bình phương gradient: G=G+(∇J)2
Bước 6: Cập nhật tham số: Sử dụng công thức sau để cập nhật từng tham số của mô hình: 
θ=θ− η /(G+ϵ ∇J)
Bước 7: Lặp lại quá trình: Lặp lại các bước trên với các mini-batch khác hoặc toàn bộ dữ liệu.
### Ưu nhược điểm và dữ liệu thích hợp
Ưu điểm:
	Tự điều chỉnh tỷ lệ học: Adagrad tự điều chỉnh tỷ lệ học cho từng tham số dựa trên lịch sử của gradient. Điều này giúp phù hợp với các tình huống trong đó các tham số có độ biến động lớn.
	Không yêu cầu lựa chọn tỷ lệ học ban đầu: Adagrad không yêu cầu người dùng phải chọn tỷ lệ học ban đầu, mà tỷ lệ này được tự động điều chỉnh dựa trên thông tin lịch sử.
	Phù hợp với các bài toán có độ chuyển động biến động lớn: Adagrad thường hoạt động tốt trên các bài toán mà các tham số có độ biến động lớn, ví dụ như các bài toán học máy sâu.
Nhược điểm:
	Độ giảm tỷ lệ học quá nhanh: Một trong nhược điểm lớn của Adagrad là sau một số bước lặp, ma trận tổng bình phương gradient có thể trở nên rất lớn, dẫn đến độ giảm tỷ lệ học nhanh chóng. Điều này có thể làm cho quá trình học trở nên chậm lại.
	Không phù hợp cho các tham số thường xuyên cập nhật: Trong một số trường hợp, khi có các tham số có gradient thường xuyên thay đổi, Adagrad có thể làm giảm quá mức tỷ lệ học, làm chậm quá trình học.
Dữ liệu thích hợp: dữ liệu với các tham số có độ biến động lớn và thường xuyên thay đổi; dữ liệu thưa, nơi một số thành phần của gradient có giá trị lớn, trong khi nhiều thành phần khác có giá trị gần 0.
## Adadelta
### Nguyên lí
Adadelta được thiết kế để giảm nhược điểm của Adagrad liên quan đến giảm tỷ lệ học quá nhanh do tích lũy bình phương gradient. Nguyên lý cơ bản của Adadelta là sử dụng trung bình trượt có trọng số của bình phương gradient để điều chỉnh tỷ lệ học
Các bước thực hiện Adadelta:
Bước 1: Khởi tạo các tham số: Khởi tạo các tham số của mô hình, bao gồm các trọng số (θ) và các tham số khác.
Bước 2: Khởi tạo ma trận trung bình trượt bình phương gradient (E[g2]) và ma trận trung bình trượt bình phương của thay đổi tham số (E[Δθ2]) bằng 0 hoặc giá trị khác nhau tùy thuộc vào chiến lược khởi tạo.
Bước 3: Thiết lập hệ số giảm trọng số (ρ): Đặt giá trị cho hệ số giảm trọng số (ρ).
Bước 4: Lặp qua từng mini-batch hoặc toàn bộ dữ liệu: Tính gradient (∇J) của hàm mất mát đối với từng tham số của mô hình.
Bước 5: Cập nhật ma trận trung bình trượt bình phương gradient (E[g2]):
Sử dụng công thức sau để cập nhật E[g2]: 
E[g2]=ρE[g2]+(1−ρ)(∇J)2
Bước 6: Tính toán thay đổi tham số (Δθ):
Tính toán thay đổi tham số (Δθ) bằng tỷ lệ của tham số gradient và căn bậc hai của ma trận trung bình trượt bình phương của thay đổi tham số: 
Δθ = −√(E[g2]+ϵ)/√(E[Δθ2]+ϵ) ∇J
Trong đó ϵ là một giá trị nhỏ để tránh chia cho 0.
Bước 7: Cập nhật tham số (θ): Sử dụng công thức: θ=θ+Δθ
Bước 8: Cập nhật ma trận trung bình trượt bình phương của thay đổi tham số (E[Δθ2]): Sử dụng công thức E[Δθ2]: E[Δθ2]=ρE[Δθ2]+(1−ρ)(Δθ)2
Bước 9: Lặp lại các bước trên với các mini-batch khác hoặc toàn bộ dữ liệu.

### Ưu nhược điểm và dữ liệu thích hợp
Ưu điểm
	Tự điều chỉnh tỷ lệ học: Adadelta tự điều chỉnh tỷ lệ học cho từng tham số một cách tự động dựa trên lịch sử của gradient, giúp tránh được vấn đề giảm tỷ lệ học quá nhanh như trong Adagrad.
	Không yêu cầu lựa chọn tỷ lệ học ban đầu: Khác với một số thuật toán khác, Adadelta không yêu cầu người dùng phải lựa chọn tỷ lệ học ban đầu.
	Ít tham số cần tinh chỉnh: Adadelta có ít tham số cần tinh chỉnh so với một số thuật toán khác, giúp giảm công đoạn lựa chọn tham số.
Nhược điểm:
	Không hoạt động tốt trên mọi bài toán: Mặc dù Adadelta hoạt động tốt trên nhiều loại bài toán, nhưng không phải lúc nào cũng là lựa chọn tốt nhất. Có những trường hợp nơi các thuật toán khác như Adam có thể mang lại hiệu suất tốt hơn.
	Không thích hợp cho bài toán có tham số thường xuyên thay đổi: Adadelta có thể không hiệu quả trên các bài toán mà tham số thường xuyên thay đổi đột ngột, do quá trình điều chỉnh tỷ lệ học của nó cũng diễn ra tương đối chậm.
Dữ liệu thích hợp: Adadelta thường hoạt động tốt trên các tập dữ liệu không đồng nhất và những tập dữ liệu ít biến động trong thời gian ngắn.
## So sánh các phương pháp
| Phương Pháp	| Độ Hiệu Quả	| Bộ Dữ Liệu Phù Hợp |	Ưu Điểm |	Nhược Điểm |
|-------------|-------------|--------------------|----------|------------|
| GD |	Trung bình |	Lớn |	Dễ hiểu, hội tụ ổn định |	Chậm trên dữ liệu lớn, nhạy cảm với nhiễu |
| BGD |	Thấp |	Nhỏ hoặc Lớn |	Hội tụ chính xác, ít nhạy cảm |	Chậm trên dữ liệu lớn, có thể rơi vào điểm tối ưu cục bộ |
| SGD |	Cao |	Nhỏ |	Hội tụ nhanh, phù hợp với dữ liệu lớn |	Không ổn định, nhảy số lớn, nhạy cảm với nhiễu |
| RMSprop |	Cao	Mọi loại |	Hiệu quả trên dữ liệu không đồng nhất |	Cần lựa chọn tham số, nhạy cảm với outlier |
| Momentum |	Cao |	Mọi loại |	Giảm ổn định của GD, vượt qua điểm tối ưu cục bộ |	Cần lựa chọn tham số, nhạy cảm với outlier |
| Adam |	Cao |	Mọi loại |	Hiệu quả và linh hoạt, tự điều chỉnh tỷ lệ học |	Cần lựa chọn tham số, yêu cầu bộ nhớ lớn |
| Adagrad |	Trung bình |	Mọi loại |	Tự điều chỉnh tỷ lệ học, ít tham số cần tinh chỉnh |	Độ giảm tỷ lệ học quá nhanh, không phù hợp cho dữ liệu thưa |
