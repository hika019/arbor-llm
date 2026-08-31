# configs/arbor_1b_8k.yaml training samples after carry-over packing

実効設定: configs/arbor_1b_8k.yaml の data 設定に train.py と同じく speed.micro_batch_size / seed を反映。shuffle_buffer は設定値のまま。

各サンプルは input_ids 8192個分を UTF-8 byte に戻した表示です。特殊tokenは <EOS> / <PAD> として表示しています。

注: document packing は短文の余りを次サンプルへ持ち越す修正後の挙動です。

## 日本語 source samples

### 日本語 source samples #1

- batch_index: 0
- row: 1
- source_id: 0
- source.id: fineweb2_ja
- source.path: HuggingFaceFW/fineweb-2
- source.name: jpn_Jpan
- weight_bytes: 0.39
- context: 8192
- fill_ratio: 1.000000
- EOS count: 1
- PAD count: 0
- masked_labels: 1

<SAMPLE_TEXT_BEGIN>
濃いのにスッキリというバランスがおもしろい
通常のオランジーナも改良したポイントがわかります
さっそく飲んでみると、やはりオレンジの香り、味わいが口の中にグッと広がります。しかし、後味にはグレープフルーツも効いていますし、ぶどうの甘みとレモンの酸味でスッキリ感もあります。果実感の濃さと、さわやかな飲み口を両立しているところがおもしろい。
オレンジピールの苦味もほのかにあり、全体としてはたしかにオレンジの風味が濃いのですが、ぶどうやレモンをたくみにブレンドして、ピールのえぐみをうまく抑えることに成功しています。しっかりフルーティーでありながら、けっしてくどくない。なかなかバランスの取れた飲みごたえです。
なるほど、オランジーナ100のおいしさはわかりました。ではリニューアルしたとうたう、通常のオランジーナはどうでしょう？
こちらは、オレンジを丸かじりしたようなジューシーな果実感と自然なピールの味わいを強化し、果実感をより感じられる厚みのある中味を実現した……とのこと。飲んでみると、ややオレンジの風味が強くなった感があります。甘みも増したでしょうか。果汁12%ながら、オレンジの味や香りが感じられる改良になっていると感じました。
果汁100%をかかげながらオレンジ以外の果汁をブレンドすることで、果実感とスッキリ感の両立をめざしたオランジーナ100。よりオレンジの雰囲気を増した通常のオランジーナ。どちらもメーカーの意気込みが感じられる味わいに仕上がっています。暑くなってくるこれからの時期、ぜひチェックしてみてください。
モーダル小嶋
1986年生まれ。担当分野は「なるべく広く」のオールドルーキー。編集部では若手ともベテランともいえない微妙な位置。
アスキーでは楽しいグルメ情報を配信しています。新発売のグルメネタ、オトクなキャンペーン、食いしんぼ記者の食レポなどなど。コチラのページにグルメ記事がまとまっています。ぜひ見てくださいね！
週刊アスキーの最新情報を購読しよう
<EOS>
洗練されたデザインとプレイヤーの創造力を刺激する独創的なサウンドで、ブランド創立から数年で世界から注目されるブランドへと成長を遂げたGamechanger Audio（ゲームチェンジャー・オーディオ）。シーンに新たな価値観を提示してみせたユニークな製品はどのように生み出されたのだろうか？ 革新性と狂気の知性を併せ持つ気鋭のエフェクター・ブランドの魅力を紐解いていこう。
解説＝今井靖 編集＝尾藤雅哉 撮影＝星野俊
※本記事はギター・マガジン2021年11月号に掲載された『GAMECHANGER AUDIO 未来のスタンダードを書き換える革新と狂気のインテリジェンス』を再編集したものです。
HISTORY
飽和したエフェクター産業への強烈なカウンター・アクション
設立後からわずか数年の間に、世界から熱い注目を集めるまでに成長したGamechanger Audio。ここでは未知なる可能性にチャレンジし続けるブランドの歩みを振り返ってみよう。
高度な技術と前衛的な音色、造形で魅せるアート・ギミック
“この世にあるほとんどのギター機材は「退屈」だね”──確信を込めてそう言い放つイリヤ・クレメンスからすれば、自らが立ち上げたGamechanger Audioの世界的な成功ですら、積み上げたロジックの単なる帰趨（きすう）に過ぎないのだろう。
高度なエレクトロニクス、アバンギャルド・サウンド、そして造形で魅せる優美なアート・ギミック。それらすべてを完璧に整合したまったく新しい価値への探求は、この刺激的なギア・メーカーにおいて創業時から変わらぬたったひとつの推進力である。
ブランドのプロダクト・デザイナーであるイリヤ・クレメンスは、幼少期をラトビア共和国の首都リガで過ごした。旧共産圏から独立して間もない母国に新たに広まった西側カルチャーの中でも、特にカントリーやロカビリー、ロックンロールといったアメリカン・サウンドに強く傾倒し、16歳になる頃には地元のロカビリー・バンドでプロとして演奏をするレベルになっていたという。
やがて本格的に音楽を勉強するためにICMP（The Institute of Contemporary MusicPerformance）ロンドン校へ渡った彼は、そこでステージ経験もほとんどない海外の若いギタリストたちが複数のストンプ・ペダルで音作りに奔走していたことを知る。
ペダル文化圏のトラディショナルな価値観をあえて否定はしなかったものの、そうした機器の多くは初歩的な電子知識さえあれば簡単に構築できるものであり、似たり寄ったりのその見た目や、それらがもたらす“わずかなサウンドの差”を問題視する傾向に触れるたびに、彼の反骨心はクリエイティブな衝動を禁じ得なくなっていった。
世界にインパクトを与えた斬新な製品たち
2015年、故郷のリガへ戻ったイリアはICMP在学中に蓄えたアイデアを形にするために、エレクトロニクス分野に精通した仲間を集め始める。最初に意気投合したのはメカニカル・エンジニアとして実績のある技術者、クリスタプス・カルバであった。続いて、クリスタプスの友人で経験豊富な回路設計者であるマーティンス・メルスキがチームに加わり、プロジェクトは本格的に動き出した。
3人が目指したのは、“形状からそれがどんな効果をもたらすのか直感的に理解でき、そこからさらにインスピレーションを与えることのできるデバイス”の具現化だった。最初のプロトタイプが完成間際になった同年9月、4人目の仲間となる営業・財務担当のディジス・ドゥボスキースが加わり、彼らはついに自らのブランドGamechanger Audioを発足させる。
そこからさらに試行錯誤を重ね、2017年のNAMM Showで大々的にお披露目されたその画期的なサンプリング・サスティナー・ユニットであるPLUS Pedalは、ピアノをイメージさせる真鍮製フット・バーの美しさ、正確かつ自由度の高いループ・テクノロジー、あらゆる創造性を刺激する音楽的なエフェクト・スプリードのすべてを兼ね備え、「Gamechanger（形勢逆転）」の名に恥じないインパクトとともに全世界に彼らの偉業を知らしめることとなった。
クラウドファンディングによって早々にPLUS Pedalの量産化の目処をつけた彼らは、その年の夏のNAMMの帰りに立ち寄ったケンタッキー州のゲスト・ハウスで、さらなる斬新なペダルのアイデアに遭遇する。そこに設置されていた電撃殺虫灯の耳慣れないスパーク・ノイズに触発されたクリスタプスが、高電圧なキセノン管の放電ノイズをディストーション・サウンドに利用したペダルを思いついたのはまさに僥倖（ぎょうこう）であった。
翌2018年、PLASMA Pedalと名付けられたその未知の機構を持つ歪みペダルは、リリースと同時にブランドにかつてない富と名声をもたらしただけでなく、彼らのもうひとつの夢をも実現させることとなる。それは、かねてより敬�
<SAMPLE_TEXT_END>

### 日本語 source samples #2

- batch_index: 1
- row: 0
- source_id: 1
- source.id: wikipedia_ja
- source.path: wikimedia/wikipedia
- source.name: 20231101.ja
- weight_bytes: 0.17
- context: 8192
- fill_ratio: 1.000000
- EOS count: 0
- PAD count: 0
- masked_labels: 0

<SAMPLE_TEXT_BEGIN>
宮本 輝紀（みやもと てるき、1940年12月26日 - 2000年2月2日）は、広島県広島市宇品（現・南区宇品）出身（広島市段原山崎町生まれ）の元サッカー選手（MF）・コーチ・監督。日本代表の攻撃的ミッドフィールダーとしてメキシコ五輪銅メダル獲得に貢献し、日本最初のゲームメーカーとも呼ばれる。

来歴
1945年8月6日、4歳の夏に爆心地から約2kmの段原山崎町で被爆し、一緒に遊んでいた弟を亡くした。張本勲も近所で被爆している。終戦後は宇品に引越して広島市立千田小学校に入学し、千田小の同期に岡光龍三、一学年上に後に山陽高→八幡→新日鐵と同じ道を歩む大石信幸がいた。広島市立国泰寺中学校時代に大学生にサッカーを教えてもらったことがきっかけでサッカー部に転部し、国泰寺中の一学年上に野村六彦、同期に今西和男がいた。宮本と野村は、後の1960年代に「日本を代表する二人のテクニシャン」と称されて誰もが認める存在となるが、その源流は国泰寺中にあった。国泰寺中サッカー部は、当時今西が入部できないほど希望者が多い状況であったが、その中でも宮本の才能は際立っており、広大付属の桑田隆幸とともに地元では有名な選手となっていた。国泰寺中は隣接する国泰寺高校の付属校ではないが、OBが多数進んでいたことから、レギュラークラスを全国有数の名門である国泰寺高校の練習に参加させた。今でいう、Jクラブの一貫教育を昭和30年代に、それも全国レベルの選手たちによって体験させていた。また、国泰寺高校では「全広島対全関西」などの試合が行われ、全日本選手のプレーを身近に見る機会があった。

中学卒業後は野村が進んだ舟入高校へ行く予定であったが、受験制度が変わり、宮本の住む地域からは入りづらくなったため、大石が進んだ創部3年目の新興勢力・広島山陽高校に進学。渡部英麿の厳しい指導を受け、2年次の1957年と3年次の1958年には国体準優勝を果たし、特に1958年は2年連続で決勝対決となった杉山隆一のいた静岡代表・清水東との雨中の死闘は有名である。

広島高師出身で東福岡高コーチとしても知られる名将・寺西忠成監督の目に留まり、寺西からの熱心な要望により1959年に八幡製鉄へ入部。寺西は広島一中で渡部の1年後輩にあたり旧知の間柄であった関係から、当時の八幡は山陽の一学年先輩の大石をはじめ、主力は広島出身者であった。1959年はクアラルンプールで開催された第1回アジアユースサッカー日本代表にも選出されて3位に貢献し、1960年には19歳11ヵ月でA代表入りを果たす。八幡でもエースとして活躍し、1963年と1964年には全日本実業団選手権2連覇、1964年の天皇杯では古河電工との両チーム優勝に導く。1965年から始まったJSLでも主力選手としてチームを引っ張り、対戦相手はまず宮本をどう抑えるかに苦心した。初年度から2年連続2位の好成績を挙げるなど通算139試合に出場し、通算68得点は歴代6位にランクインしている。この記録は年間14試合しか行われていない時代に残した記録であり、歴代でも上位を争う価値のあるものである。ベストイレブンには6度選出されているが、八幡は社業の悪化で、JSLが発足した1965年直後から新人補強で苦戦。ライバルチームとの差が開き、この後はチームとしてのタイトル獲得はならなかった。1967年には日本年間最優秀選手賞にも選ばれ、1970年にはJSLアシスト王に輝いた。代表では1960年代から1970年代にかけて国内最高のテクニシャンで、そのテクニックは当時の代表の中でも群を抜いていた。東京五輪から代表の頭脳となってゲームを組み立てるようになり、「天才パサー」と呼ばれた。パス1本で相手を窮地に追い込む「元祖キラーパス」は、後の代表司令塔・中田英寿、中村俊輔をも凌ぐと評される。前線の釜本邦茂や杉山にパスを供給するのが日本の攻撃パターンであり、当時「パワーの釜本、スピードの杉山、テクニックの宮本」と呼ばれたトライアングルはサッカー選手を志す少年達の憧れであった。表に出ることが好きでない性格で、派手なパフォーマンスは嫌いで口数も少なく、自らゴールを決めた直後も周囲が歓喜する中でつまらなそうにペッと唾を吐き、一人憮然としていたといわれる。北九州市出身の本間勇輔も大ファンだったと話しているほか、後藤健生や国吉好弘らも宮本のプレーを見て感銘を受けたのが、サッカージャーナリストになったきっかけと話している。パスもさる事ながらゴールにも迫りシュートも連発し、振り幅の小さいシュートでゴールを量産してMFながら国際Aマッチ18得点は歴代8位、代表での全試合では歴代4位の47得点（出場192試合、歴代3位）を挙げている。いずれもメキシコ五輪世代では釜本に次ぐ数字であり、そのメキシコ五輪では、中盤の守備の要で主将の八重樫茂生が初戦で負傷。そのため宮本は司令塔でありながら八重樫の代役も兼ねたが、パスを出しながら必死に守備をして縦横無尽に走り回り、メキシコとの3位決定戦では精根尽き果て倒れ込んだ。1974年から1975年には選手兼任コーチとして渡辺正監督を支え、渡辺が総監督となった1976年からは渡辺の後任でプレイングマネージャーとなり、1年目のリーグ戦は9位に終わるが、同年の天皇杯でベスト4に導く。1部・2部入替戦では読売クラブの昇格を阻んで残留を決め、現役を引退。

引退後も新日鐵の監督（1977年 - 1979年）を務め、オイルショック後でさらに補強が厳しくチームは低迷したが、JSLカップでは1977年にベスト8、1978年にベスト4と好成績を残す。派手嫌いで実直な人柄で知られ、選手として頂点を極めて引退した後も勤務地である北九州にとどまった。中央に出てくることは無く、国体福岡県代表監督（1981年 - 1985年）・九州共立大学監督（1996年 - 1999年）を務め、九州共立大学では僅か2年で九州大学リーグ1部に昇格させた。1993年からスタートしたJリーグに新日鐵は、地域性から参加を要請されたが参加しなかった。

2000年2月2日、八幡東区の病院で心不全のため死去。59歳没。2006年に高卒の選手経験者では最初の日本サッカー殿堂入りを果たし、母校・山陽高校の正門に入ると右手に宮本の功績を讃える記念碑がある。

所属クラブ
 1959年 - 1976年 八幡製鉄/新日本製鐵

個人成績

!colspan="4"|日本!!colspan="2"|リーグ戦!!colspan="2"|JSL杯!!colspan="2"|天皇杯!!colspan="2"|期間通算
|-
|1965||rowspan=5|八幡||||rowspan=7|JSL||||7||colspan="2"|-||||||||
|-
|1966||||||11||colspan="2"|-||||||||
|-
|1967||||||10||colspan="2"|-||colspan="2"|-||||
|-
|1968||||||11||colspan="2"|-||||||||
|-
|1969||||||7||colspan="2"|-||||||||
|-
|1970||rowspan=7|新日鐵||||||6||colspan="2"|-||colspan="2"|-||||
|-
|1971||||||7||colspan="2"|-||||||||
|-
|1972||||rowspan=5|JSL1部||||||colspan="2"|-||||||||
|-
|1973||||||||||||||||||
|-
|1974||||||||colspan="2"|-||||||||
|-
|1975||||||||colspan="2"|-||||||||
|-
|1976||||||||||||||||||
138||68||||||||||||
138||68||||||||||||
|}

代表歴
 日本�
<SAMPLE_TEXT_END>

### 日本語 source samples #3

- batch_index: 1
- row: 1
- source_id: 2
- source.id: aozora_modern
- source.path: globis-university/aozorabunko-clean
- source.name: None
- weight_bytes: 0.02
- context: 8192
- fill_ratio: 1.000000
- EOS count: 0
- PAD count: 0
- masked_labels: 0

<SAMPLE_TEXT_BEGIN>
　　　　　　　一

　天保二年九月の或午前である。神田同朋町の銭湯松の湯では、朝から不相変客が多かつた。式亭三馬が何年か前に出版した滑稽本の中で、「神祇、釈教、恋、無常、みないりごみの浮世風呂」と云つた光景は、今もその頃と変りはない。風呂の中で歌祭文を唄つてゐる嚊たばね、上り場で手拭をしぼつてゐるちよん髷本多、文身の背中を流させてゐる丸額の大銀杏、さつきから顔ばかり洗つてゐる由兵衛奴、水槽の前に腰を据ゑて、しきりに水をかぶつてゐる坊主頭、竹の手桶と焼物の金魚とで、余念なく遊んでゐる虻蜂蜻蛉、――狭い流しにはさう云ふ種々雑多な人間がいづれも濡れた体を滑らかに光らせながら、濛々と立上る湯煙と窓からさす朝日の光との中に、糢糊として動いてゐる。その又騒ぎが、一通りではない。第一に湯を使ふ音や桶を動かす音がする。それから話し声や唄の声がする。最後に時々番台で鳴らす拍子木の音がする。だから柘榴口の内外は、すべてがまるで戦場のやうに騒々しい。そこへ暖簾をくぐつて、商人が来る。物貰ひが来る。客の出入りは勿論あつた。その混雑の中に――
　つつましく隅へ寄つて、その混雑の中に、静に垢を落してゐる、六十あまりの老人が一人あつた。年の頃は六十を越してゐよう。鬢の毛が見苦しく黄ばんだ上に、眼も少し悪いらしい。が、痩せてはゐるものの骨組みのしつかりした、寧いかついと云ふ体格で、皮のたるんだ手や足にも、どこかまだ老年に抵抗する底力が残つてゐる。これは顔でも同じ事で、下顎骨の張つた頬のあたりや、稍大きい口の周囲に、旺盛な動物的精力が、恐ろしい閃きを見せてゐる事は、殆壮年の昔と変りがない。
　老人は丁寧に上半身の垢を落してしまふと、止め桶の湯も浴びずに、今度は下半身を洗ひはじめた。が、黒い垢すりの甲斐絹が何度となく上をこすつても、脂気の抜けた、小皺の多い皮膚からは、垢と云ふ程の垢も出て来ない。それがふと秋らしい寂しい気を起させたのであらう。老人は片々の足を洗つたばかりで、急に力がぬけたやうに手拭の手を止めてしまつた。さうして、濁つた止め桶の湯に、鮮かに映つてゐる窓の外の空へ眼を落した。そこには又赤い柿の実が、瓦屋根の一角を下に見ながら、疎に透いた枝を綴つてゐる。
　老人の心には、この時「死」の影がさしたのである。が、その「死」は、嘗て彼を脅したそれのやうに、忌はしい何物をも蔵してゐない。云はばこの桶の中の空のやうに、静ながら慕はしい、安らかな寂滅の意識であつた。一切の塵労を脱して、その「死」の中に眠る事が出来たならば――無心の子供のやうに夢もなく眠る事が出来たならば、どんなに悦ばしい事であらう。自分は生活に疲れてゐるばかりではない。何十年来、絶え間ない創作の苦しみにも、疲れてゐる。……
　老人は憮然として、眼を挙げた。あたりではやはり賑な談笑の声につれて、大ぜいの裸の人間が、目まぐるしく湯気の中に動いてゐる。柘榴口の中の歌祭文にも、めりやすやよしこのの声が加はつた。ここには勿論、今彼の心に影を落した悠久なものの姿は、微塵もない。
「いや、先生、こりやとんだ所で御眼にかかりますな。どうも曲亭先生が朝湯にお出でにならうなんぞとは手前夢にも思ひませんでした。」
　老人は、突然かう呼びかける声に驚ろかされた。見ると彼の傍には、血色のいい、中背の細銀杏が、止め桶を前に控へながら、濡れ手拭を肩へかけて、元気よく笑つてゐる。これは風呂から出て、丁度上り湯を使はうとした所らしい。
「不相変御機嫌で結構だね。」
　馬琴滝沢瑣吉は、微笑しながら、稍皮肉にかう答へた。

　　　　　　　二

「どう致しまして、一向結構ぢやございません。結構と云や、先生、八犬伝は愈出でて、愈奇なり、結構なお出来でございますな。」
　細銀杏は肩の手拭を桶の中へ入れながら、一調子張上げて弁じ出した。
「船虫が瞽婦に身をやつして、小文吾を殺さうとする。それが一旦つかまつて拷問された揚句に、荘介に助けられる。あの段どりが実に何とも申されません。さうしてそれが又、荘介小文吾再会の機縁になるのでございますからな。不肖ぢやございますが、この近江屋平吉も、小間物屋こそ致して居りますが、読本にかけちや一かど通のつもりでございます。その手前でさへ、先生の八犬伝には、何とも批の打ちやうがございません。いや全く恐れ入りました。」
　馬琴は黙つて又、足を洗ひ出した。彼は勿論彼の著作の愛読者に対しては、昔からそれ相当な好意を持つてゐる。しかしその好意の為に、相手の人物に対する評価が、変化するなどと云ふ事は少しもない。これは聡明な彼にとつて、当然すぎる程当然な事である、が、不思議な事には逆にその評価が彼の好意に影響すると云ふ事も亦殆どない。だから彼は場合によつて、軽蔑と好意とを、完く同一人に対して同時に感ずる事が出来た。この近江屋平吉の如きは、正にさう云ふ愛読者の一人である。
「何しろあれだけのものをお書きになるんぢや、並大抵なお骨折ぢやございますまい。先づ当今では、先生がさしづめ日本の羅貫中と云ふ所でございますな――いや、これはとんだ失礼を申上げました。」
　平吉は又大きな声をあげて笑つた。その声に驚かされたのであらう。側で湯を浴びてゐた小柄な、色の黒い、眇の小銀杏が、振返つて平吉と馬琴とを見比べると、妙な顔をして流しへ痰を吐いた。
「貴公は不相変発句にお凝りかね。」
　馬琴は巧に話頭を転換した。がこれは何も眇の表情を気にした訳ではない。彼の視力は幸福な事に（？）もうそれがはつきりとは見えない程、衰弱してゐたのである。
「これはお尋ねに預つて恐縮至極でございますな。手前のはほんの下手の横好きで今日も運座、明日も運座、と、所々方々へ臆面もなくしやしやり出ますが、どう云ふものか、句の方は一向頭を出してくれません。時に先生は、如何でございますな、歌とか発句とか申すものは、格別お好みになりませんか。」
「いや私は、どうもああ云ふものにかけると、とんと無器用でね。尤も一時はやつた事もあるが。」
「そりや御冗談で。」
「いや、完く性に合はないとみえて、未だにとんと眼くらの垣覗きさ。」
　馬琴は、「性に合はない」と云ふ語に、殊に力を入れてかう云つた。彼は歌や発句が作れないとは思つてゐない。だから勿論その方面の理解にも、乏しくないと云ふ自信がある。が、彼はさう云ふ種類の芸術には、昔から一種の軽蔑を持つてゐた。何故かと云ふと、歌にしても、発句にしても、彼の全部をその中に注ぎこむ為には、余りに形式が小さすぎる。だから如何に巧に詠みこなしてあつても、一句一首の中に表現されたものは、抒情なり叙景なり、僅に彼の作品の何行
<SAMPLE_TEXT_END>

## 英語 source samples

### 英語 source samples #1

- batch_index: 0
- row: 0
- source_id: 4
- source.id: fineweb_edu_en
- source.path: HuggingFaceFW/fineweb-edu
- source.name: None
- weight_bytes: 0.12
- context: 8192
- fill_ratio: 1.000000
- EOS count: 4
- PAD count: 0
- masked_labels: 4

<SAMPLE_TEXT_BEGIN>
Monuments and Memorials
Throughout the 639 acres of Arlington National Cemetery, several dozen monuments and memorials commemorate individuals, military units, wars and battles. These commemorative works, many of which were placed in the late 19th and early 20th centuries, both reflected and furthered Arlington’s transformation from a “pauper’s cemetery” to a national shrine. Several are the works of renowned artists: noted early 20th-century sculptor Frances Rich created the art deco Nurses Memorial, for example (pictured); Edward Clark Potter, best known for the New York Public Library’s iconic marble lions, designed the equestrian Kearny Memorial. The diverse styles of these commemorative works also reflect shifting trends in cemetery and memorial design, from the classical and naturalistic styles of the 19th-century rural cemetery movement to more contemporary geometric designs that harmonize with the cemetery’s rows of white marble headstones.
Throughout the course of Arlington National Cemetery’s history, policies on the placement of commemorative works have changed, in response to the changing needs of the cemetery. Under current federal law (38 USC §2409), commemorative monuments may be placed on ANC grounds only after a deliberative proposal process. Monuments that do not contain or mark interred remains may only be approved for placement if they meet the criteria specified by 38 USC §2409. Click here for more information.
<EOS>
Date of Award
Master of Science in Geology
Reinhard K. Frohlich
The causes of the seismicity of the St. Lawrence River Valley are not well understood. As is the case for the entire east coast of North America, epicentral zones often occur in regions where no correlation exists between seismicity and mapped geologic structures. There are several explanations for such a phenomenon: a) earthquakes occur along unmapped surface faults, b) earthquakes occur along subsurface faults showing no surface expression, or c) the structures are not fault related.
Conventional filtering techniques, such as the upward continuation, downward continuation and second derivative methods were applied to gravity data from the St. Lawrence River Valley in an attempt to delineate possible seismic-related structures. The gravity survey indicates that the anomalies trend in a north-northeast direction similar to the structural trends of the Precambrian rocks. The major feature of the Simple Bouguer map is an extensive gravity high centered at Massena, New York.
Analyses by the filtering methods and subsequent modeling by the two-dimensional Talwani technique reveal the existence of two anomaly-producing bodies responsible for the Massena High: 1) a wedge (8x35 km) located 6 km below sea-level with a density contrast of +.11g/cc, and 2) a smaller body (2x6 km) located 3.3km below sea level with a density contrast of +.2g/cc. The large wedge may represent a sequence of interlayered metasesiments and meavolcanics related to the Grenville sequence. The smaller body may represent a mafic intrusive. In addition, the Simple Bouruer map reveals a circular gravity of low intensity (diameter=20km) centered at Cornwall, Ontario.
The Cornwall Low suggests an association of high gradients of gravity (toward positive) produced by mafic intrusives and earthquakes in the southeastern United States. The possible existence of a mafic intrusive near Massena, New York and its proximity to epicentral zones suggests a similar origin for some earthquakes in the study area.
Albert, Robert Louis, "A Gravity Study of Earthquake-Related Structures in the St. Lawrence River Valley" (1977). Open Access Master's Theses. Paper 990.
<EOS>
Your Family Earthquake Preparedness Plan includes recording information to facilitate communication among family members, and to be aware that during the first 24 hours after an earthquake, the phone should only be used for emergencies. Text messaging may be an alternative option before phone calls are available.
Download and print these emergency documents while you have power and access to the internet.
Click the link to open the PDF in a new tab, or click the download button to store locally on your device.
According to the USGS, Southern California has about 10,000 earthquakes per year, most are mild and we don’t even feel them. Living along the San Andreas fault, we have the possibility of experiencing a destructive quake.
Earthquakes and fires can release asbestos fibers which are extremely dangerous. Learn more about asbestos in natural disasters.
Protecting your children, pets, and more
The Great California Shake Out
International ShakeOut Day is always the third Thursday of October.
As always, you can hold your ShakeOut drill when and where you want. You can choose another date or several dates, and include people in multiple locations (home, work, or school), perhaps through video conferencing. It’s actually a good idea to practice earthquake safety in different situations each year!
Visit our full library of resources.
Get on our Calabasas mailing list.
Watch our video collection.
<EOS>
A cookie is a small file that holds a certain amount of data, that are downloaded to your computer, to improve your experience. Cookies are often used to remember information about preferences and pages a person has visited. This cookie data can then be retrieved and can allow us to customise our web pages and services accordingly. It is important to clarify that cookies do not collect any personal data stored on your hard drive or computer.
The cookies we use do not store personally identifiable information nor can they harm your computer. We want our website to be informative, and as user-friendly as possible, and cookies help us to achieve that goal.
3. Social Media Third Party Cookies
4. How to Control and Delete Cookies
You can prevent the setting of cookies by adjusting the settings on your browser (see your browser Help to do so)
<EOS>
Best Practices in System Design
Introduction to System Design: Best Practices
System design is a fundamental aspect of software development that involves planning and organizing the architecture of a software system. It plays a crucial role in the development process as it determines how effectively the system will perform, scale, and maintain. In this tutorial, we will discuss the best practices in system design to help you create robust and scalable software systems.
Understanding the Requirements
Before diving into the design process, it is essential to understand the requirements thoroughly. Start by gathering functional and non-functional requirements from stakeholders, users, and other relevant sources. Documenting these requirements helps in creating a clear roadmap for the system design.
Identify Key Components
Once you have a clear understanding of the requirements, start by identifying the key components of the system. Break down the system into smaller modules or services that can be designed and implemented separately. This modular approach makes the system more manageable and scalable in the long run.
Design Patterns and Architectural Styles
A good system design incorporates well-established design patterns and architectural styles. These patterns provide proven solutions to common design problems, making the system more maintainable and extensible. Examples of popular design patterns include:
MVC Pattern (Model-View-Controller)
The Model-View-Controller pattern separates the application's data (model) from its presentation (view) and the user interaction logic (controller). It enhances flexibility and code reusability, making the system easier to maintain and extend.
Microservices architecture involves building an application as a collection of small, loosely coupled services. Each service focuses on a specific business capability and can be developed, deployed, and scaled independently. This architectural style offers benefits such as better scalability, fault isolation, and resilience.
Scalability and Performance
Designing a scalable and performant system is crucial for handling increasing workloads and ensuring optimal user experience. Consider the following best practices to achiev
<SAMPLE_TEXT_END>

### 英語 source samples #2

- batch_index: 2
- row: 1
- source_id: 5
- source.id: fineweb_en
- source.path: HuggingFaceFW/fineweb
- source.name: None
- weight_bytes: 0.05
- context: 8192
- fill_ratio: 1.000000
- EOS count: 2
- PAD count: 0
- masked_labels: 2

<SAMPLE_TEXT_BEGIN>
Fabric Specification: 100% Cotton
Description: Step back in time with the Vintage Charcoal Men's Baggy Jeans. These jeans offer a relaxed, retro fit with a roomy leg and a high-rise waist for ultimate comfort. The vintage charcoal wash gives them a classic, worn-in look, perfect for adding a touch of nostalgia to your wardrobe. Crafted from durable denim, they combine style and practicality, making them ideal for casual outings or laid-back weekends
Products priced below ₹1000 are not eligible for return or exchange.
Products priced between ₹1000 and ₹1900 are eligible for exchange up to two times. The exchange can be for the same product or a limited selection of higher-priced products.
Products priced below ₹1900 are not eligible for return; they can only be exchanged if they fall within the ₹1000–₹1900 range.
Only products priced above ₹1900 are eligible for a single return.
Refunds for returned products will be issued in the form of a gift card only.
The eligibility criteria mentioned above are based on the individual product value and not the total cart or order value.
<EOS>
Ambode Restates Commitment to $2.3bn Badagry Port Project
The Governor of Lagos State, Mr. Akinwunmi Ambode, has said he is committed to the success of the Badagry Deep Sea Port Project, which has attracted an estimated investment of $2.3bn (N724.5bn).
Ambode also said his administration’s policies were geared towards creating a friendly environment to encourage private sector participation in driving economic development in the state, according to a statement on Friday by his Chief Press Secretary, Mr. Habib Aruna.
The governor stated this on Thursday when he met with the representatives of the Netherlands-based APM Terminals, an international container terminal operating company in London.
Saying that the Badagry Deep Sea Port Project would address the infrastructural enhancement and urban renewal agenda of the state, Ambode commended the resolve of the investors to stay the course on the project, adding that the facility would generate over 500,000 direct and indirect jobs on completion.
The governor also assured that the state would spare nothing to see to the fruition of the project, while pledging that the interests of the host communities within the location of the project would be protected.
He also said the project, on completion, would be the biggest in the African continent as it is expected to sit on a land space of over 1,000 hectares.
The Head of APM Terminals, Africa, Mr. Peter Volkjaer Jorgensen, said the company was “strongly” committed to partnering the Lagos State Government on the project.
Is the CEO and Founder of Investors King Limited. He is a seasoned foreign exchange research analyst and a published author on Yahoo Finance, Business Insider, Nasdaq, Entrepreneur.com, Investorplace, and other prominent platforms. With over two decades of experience in global financial markets, Olukoya is well-recognized in the industry.
<EOS>
Retired Lt. Gen. Michael Flynn
Date posted: November 11, 2016
An intelligence consulting firm founded by retired Lt. Gen. Michael Flynn, Donald Trump’s top military adviser, was recently hired as a lobbyist by an obscure Dutch company with ties to Turkey’s government and its president, Recep Tayyip Erdogan.
The revelation of that new lobbying contract, which has not been previously reported, raises several questions given that Trump is said to be considering Flynn, the former director of the Defense Intelligence Agency (DIA), to take over as either Secretary of Defense or National Security Advisor.
It also raises questions about disclosure.
Flynn wrote an op-ed for The Hill on Tuesday, just before Trump’s stunning upset of Hillary Clinton, in which he heaped praised on Erdogan and called on the next president, whoever that would be, to accede his request to extradite the U.S.-based Muslim cleric Fethullah Gülen back to Turkey.
“Gülen’s vast global network has all the right markings to fit the description of a dangerous sleeper terror network,” Flynn wrote in the op-ed, in which he called Gulen a “shady Islamic mullah” and “radical Islamist.”
Erdogan has accused Gülen, who has lived in exile in Pennsylvania since 1999, of masterminding a violent coup in Turkey in July. Gülen has denied doing so, but Erdogan has pressured President Obama to review evidence that the 76-year-old imam was behind the uprising, which left nearly 300 soldiers and civilians dead.
“From Turkey’s point of view, Washington is harboring Turkey’s Osama bin Laden,” Flynn asserted.
The piece does not include a disclosure that Flynn Intel Group, the consulting firm that Flynn founded in Oct. 2014, just after leaving DIA, was recently hired to lobby Congress by a Dutch company called Inovo BV that was founded by a Turkish businessman who holds a top position on Turkey’s Foreign Economic Relations Board.
A review of Dutch records shows that the company was founded by Ekim Alptekin, an ally of Erdogan’s who is director of the Turkey-U.S. Business Council, a non-profit arm of Turkey’s Foreign Economic Relations Board. Members of the Foreign Economic Relations Board are chosen by Turkey’s general assembly and its minister of economy. In the role, Alptekin helped coordinate Erdogan’s visit to the U.S. earlier this year.
A lobbying disclosure report filed with Congress lists Inovo BV’s address but not the name of anyone affiliated with the company. There is also little information about the firm online. But The Daily Caller tracked down Dutch business registration records which show Alptekin founded the company in 2005. The financial consulting firm, which Alptekin does not acknowledge on his bio, also has an affiliate, Inovo Turkije.
The lobbying disclosure does not say how much Inovo BV is paying Flynn’s firm. It lists former congressional aide Robert Kelley as the lobbyist who is handling the contract and says that he is working on “organizational consulting” for Inovo BV.
Flynn’s recent op-ed appears to be at odds with some of his past comments about Turkey and its role in the war against ISIS. In the op-ed he refers to the Islamic nation, which is a member of NATO, is “vital to U.S. interests” and is the U.S.’s “strongest ally” against ISIS.
But he told journalist Seymour Hersh for an article published earlier this year that Turkey was doing little to stop foreign fighters and weapons from crossing the border into Syria.
“We understood ISIS’s long-term strategy and its campaign plans, and we also discussed the fact that Turkey was looking the other way when it came to the growth of the Islamic State inside Syria,” Flynn told Hersh for the article.
It is unclear whether the Trump administration will side with Erdogan on the Gülen issue. The men were allies until recent years, when some of Gülen’s followers, called Gülenists, opened corruption investigations of some of Erdogan’s government allies.
Erdogan has since then assailed Gülen and his network, which he refers to as a “parallel government” because Gülen’s followers are scattered throughout Turkey’s judiciary, police force and military.
The tension peaked in July when a group of mid-level Turkish military officials attempted to overthrow the government in a battle that hit several of Turkey’s major cities, including Ankara, the capital, and Istanbul.
Erdogan immediately blamed Gülen. And though the mysterious cleric denied any involvement, Erdogan began to pressure the U.S. to return him back to his homeland to face charges pending against him. While the State Department has said it is reviewing evidence presented against Gülen, the Obama administration has appeared less than eager to extradite him.
In a statement to TheDC, Gülen’s lawyers said they hoped that Flynn’s op-ed is not indicative of the Trump administration’s position towards the cleric.
“We hope that Mr. Flynn’s op-ed on Mr. Gülen and Turkish-American relations, published before the results of the election were known, is not a statement of policy for President-Elect Trump,” Gülen’s legal team at the Washington D.C. firm
<SAMPLE_TEXT_END>

### 英語 source samples #3

- batch_index: 7
- row: 0
- source_id: 4
- source.id: fineweb_edu_en
- source.path: HuggingFaceFW/fineweb-edu
- source.name: None
- weight_bytes: 0.12
- context: 8192
- fill_ratio: 1.000000
- EOS count: 1
- PAD count: 0
- masked_labels: 1

<SAMPLE_TEXT_BEGIN>
 scalability and performance:
Implement load balancing mechanisms to distribute incoming traffic evenly across multiple servers or services. Load balancers help prevent bottlenecks and improve system reliability.
// Example load balancing code snippet in Java using a popular library
LoadBalancer lb = new RoundRobinLoadBalancer();
String server = lb.getServer();
System.out.println("Request sent to server: " + server);
Use caching mechanisms to store frequently accessed data, reducing the load on the database or expensive computations. Caching systems such as Redis or Memcached can significantly improve system response times.
# Example caching code snippet in Python using Redis
# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)
# Check if data is already in cache
data = r.get('cached_data')
print("Data retrieved from cache: " + data)
# Retrieve data from the database
data = fetch_data_from_database()
# Store data in cache for next time
print("Data retrieved from database: " + data)
A secure system design is crucial to protect user data and prevent unauthorized access. Follow these security best practices:
Authentication and Authorization
Implement strong authentication and authorization mechanisms to control user access to different parts of the system. Use industry-standard protocols such as OAuth or JWT to ensure secure user authentication.
Encrypt communication between different system components using secure protocols like HTTPS or TLS. This helps prevent data interception and ensures the integrity of transmitted data.
In this tutorial, we explored the best practices in system design, focusing on the introduction to system design and key considerations for robust and scalable systems. By following these practices, you can create software systems that are efficient, reliable, and secure.
Remember to thoroughly understand the requirements, identify key components, utilize design patterns and architectural styles, ensure scalability and performance, and prioritize security considerations. By doing so, you can design software systems that exceed expectations and provide a seamless user experience.
Hi, I'm Ada, your personal AI tutor. I can help you with any coding tutorial. Go ahead and ask me anything.
I have a question about this topic
Give more examples
<EOS>
In the tribal villages located in Moulvibazar, Bangladesh, Khasia community leaders have enforced strict restrictions so that no life is sacrificed during the pandemic. Sohanur Rahman reports.
This is the forty-fifth in the series of stories from Voices from the Frontline initiative by ICCCAD and CDKN.
Khasia (also known as Khasi) is a matriarchal ethnic group found in the Indian state of Meghalaya and north-east Bangladesh. They live a segregated life in hilly, forested villages and mostly rely on betel leaf cultivation for a livelihood. Khasias call their villages “punjis”, which are clusters of houses within the cultural boundary of their own community.
Magurchhara Punji is a tribal village inhabited by the Khasia community. It is located in Kamalganj Upazila of Moulvibazar District which is an administrative region of Northeast Sylhet division, Bangladesh. In this division, there are about 90 Khasia Punji sheltering 40,000 Khasia people. During the lockdown, the Khasia community in Magurchhara Punji has shown extraordinary skills in following health guidelines and safeguarding their locality.
Dealing with Covid-19
Gdision Prodhan Suchiang (51) is the headman (leader of the clan) of Magurchhara punji and president of a community-based organisation called “Khasi Social Council” which comprises 30 headmen from different punjis. When the first Covid-19 case of Bangladesh was identified in March, the members of the social council decided to inform their community about it.
They wanted to start off by sharing leaflets with important information and precautionary measures related to Covid-19. But since many of them do not understand Bengali language, they decided to run awareness campaigns in the local language instead.
“We have also protected our punjis by barricading entrances with traditional fences. We are running from one punji to another — advising everyone to wear masks, maintain social distancing, and adhere to hygiene rules. We ourselves maintain strict hygiene and use protective gears during the campaigns,” says Gdision.
In each punji, people have been put under strict lockdown: only those who are sick are able to leave – to get medical assistance. Residents who have been elsewhere are asked to maintain a 14-days quarantine outside before they can enter. Not even relatives of the residents have been allowed to enter during this time. Handwritten notices were posted in the punjis to warn away visitors including people from other punjis.
“Even sacks of rice and pulses bought from outside were disinfected and kept in a hut adjacent to the punji gate for a week, as a precaution against the virus. Under the strict restriction, community people have stocked three month worth of food and daily commodities in the punjis and going out was only permissible if it is for medical purposes. Piling up of goods reduced the number of visits of the vendors and whenever they visited, they were strictly told to wear masks,” he adds.
For perishable goods such as fish and vegetables, they hired two fixed vendors who would deliver the items at selected locations. After buying them, the community members were guided to wash the items properly at home. After three months when the stockpile began to decrease, each family made a list of what they would need for the next month. The volunteers collected all the lists and sent them to a shop. After receipt, the products were disinfected and kept in a secluded place for 5 days and then distributed.
From January 2021, the headmen have permitted the villagers to visit the local market but only to buy one month’s worth of supply. They have also decided to let customers enter punjis for business purposes with proper hygiene measures.
Public gatherings at religious places have also been restricted by the headmen. For the first three months, even priests and nuns who made regular pastoral visits to the village were not allowed. On Sundays, villagers held prayer gatherings in the village chapel. Recently, priests and nuns have been allowed to visit on a limited scale following health guidelines.
Following these precautionary actions, none of the villagers have been infected. Other punjis followed the same process and people have been saved from the deadly coronavirus.
Cultivation of betel leaves
Traditionally, Khasia communities grow betel-leaf on trees which is different from plain land betel-leaf cultivation. Tree-based betel-leaf cultivation is a productive and sustainable agroforestry system. Following the countrywide shutdown and business closures due to the pandemic, betel leaf cultivation was put off for a month. It created an extra financial burden on the communities, but thankfully they could recover from it after a while.
After a month of shutdown, they again start cultivating and selling betel leaves. “For this, we chose a place outside our area. There, we conducted our trading with proper hygiene protocols. After that we properly washed our hands with soap, soaked our feet in disinfectant solution and sprayed them before entering the punji,” explains Gdision.
The education of Khisia children was not much hampered compared to others, as schools used to function in two shifts, one in Khasia language and another in Bengali. As the Khasia language shift is facilitated by villagers themselves, they restarted it after being closed for two months. But the Bengali shift is still closed, in compliance with government rules.
“We suffer badly at times due to the poor condition of hilly roads and no electricity facilities. There are so many remote punjis where there are no modern facilities. People outside the community mocked us for such extreme restriction but villagers ignored them. Our main objective was to protect the community people from the pandemic by applying their own indigenous knowledge and practices. All th
<SAMPLE_TEXT_END>
